"""Keyboard-driven knife cutting a slab of raw meat, simulated with MPM + CPIC.

The meat is a Material Point Method (MPM) body sampled from the raw-meat USDZ asset, so it keeps the
asset's shape. The knife is the kitchen-knife USDZ asset, driven kinematically from the keyboard.

Cutting comes from CPIC (Compatible Particle-in-Cell): the MPM transfers skip any particle/grid-node pair
that the knife's signed distance field separates, so material on the two sides of the blade stops
exchanging momentum and the body genuinely comes apart instead of being squashed. How far the knife
descends decides the outcome: stop half way and the meat is scored but still connected underneath, go
down to the board and the two halves are severed and can be pushed apart.

Keyboard controls
-----------------
arrow keys          move the knife forwards / backwards / left / right
E / Q               raise / lower the knife
"""

import argparse
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, KDTree

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import get_cache_dir, tensor_to_array
from genesis.utils.watertighten import watertighten_mesh
from genesis.vis.keybindings import Key, KeyAction, Keybind

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
MEAT_USDZ = ASSETS_DIR / "cc0_raw_meat_4.usdz"
KNIFE_USDZ = ASSETS_DIR / "cc0_kitchen_knife.usdz"

# Local x below which knife vertices belong to the blade rather than to the handle.
BLADE_SPLIT_X = -0.012

# Surface reconstruction is substantially more expensive than one physics step. Keep the 60 Hz physics
# timestep, but only rebuild and present the SplashSurf mesh at roughly 20 Hz.
VISUAL_UPDATE_STRIDE = 3

# Cull tiny detached scraps and one-particle-wide strands twice per simulated second. The neighbourhood
# threshold includes the particle itself, so ordinary corners and cut surfaces are retained.
FRAGMENT_CLEANUP_INTERVAL = 30
MIN_FRAGMENT_PARTICLES = 12
MIN_LOCAL_NEIGHBORS = 4


def load_usd_mesh(path):
    """Return the single mesh of a USDZ asset as a trimesh, in the asset's own frame, scaled to meters.

    Only the prim's scale is applied: its rotation and translation place the asset inside its authored
    stage, which is not a frame this demo has any use for. Both assets are authored with x along their
    long axis and y up, so the caller decides how to stand them up.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    xform_cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        usd_mesh = UsdGeom.Mesh(prim)
        counts = np.array(usd_mesh.GetFaceVertexCountsAttr().Get())
        if not (counts == 3).all():
            gs.raise_exception(f"Expected a triangulated mesh in '{path}', got face sizes {np.unique(counts)}.")
        verts = np.array(usd_mesh.GetPointsAttr().Get(), dtype=np.float64)
        faces = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
        transform = np.array(xform_cache.GetLocalToWorldTransform(prim)).T
        verts = verts * np.linalg.norm(transform[:3, :3], axis=0) * meters_per_unit
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    gs.raise_exception(f"No mesh prim found in '{path}'.")


def meat_mesh(scale):
    """Return the meat as a closed, board-resting trimesh, rotated 90 degrees around vertical.

    The asset is a triangle soup of eight open shells. Signed-distance queries still classify it
    correctly, but the particle sampler and the renderer both assume a closed surface, so it is wrapped.
    """
    mesh = load_usd_mesh(MEAT_USDZ)
    # Stand the y-up asset on the board: (x, y, z) -> (x, -z, y) leaves the slab on its flat face.
    verts = mesh.vertices * scale
    verts, faces = watertighten_mesh(np.stack([verts[:, 0], -verts[:, 2], verts[:, 1]], axis=1), mesh.faces)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.vertices -= mesh.bounds.mean(axis=0)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(0.5 * np.pi, (0.0, 0.0, 1.0)))
    return mesh


def blade_mesh(edge_thickness, spine_thickness):
    """Return the knife's collision blade: the authored blade outline extruded into a tapered wedge.

    The authored blade is a 1.4 mm plate assembled from open shells, so it has no usable signed distance
    field and the kerf it carves is far narrower than an MPM cell, which lets the two sides merge back
    together the moment the knife leaves. This wedge stands in for collision only, thin at the edge so it
    still parts the meat like a blade and thick at the spine so the cut it opens survives; the knife the
    user sees stays the authored mesh at its authored thickness.

    Built in the asset frame, so it shares the geom-local axes of the visual knife.
    """
    # Asset frame: x runs tip to handle, y is the blade thickness, z is the blade width with the sharp
    # edge at -z. The outline is taken in the x-z plane, where the blade is a flat profile, and its
    # convex hull is the blade shape itself, a straight spine over a curved edge. Coincident points are
    # dropped first: the hull would keep both copies, and the duplicate vertices they seed in the wedge
    # make it non-manifold as soon as the collision-mesh loader welds them.
    profile = load_usd_mesh(KNIFE_USDZ).vertices
    profile = np.unique(profile[profile[:, 0] < BLADE_SPLIT_X][:, [0, 2]], axis=0)
    outline = profile[ConvexHull(profile).vertices]

    z_edge, z_spine = outline[:, 1].min(), outline[:, 1].max()
    grind = (outline[:, 1] - z_edge) / (z_spine - z_edge)
    half_thickness = 0.5 * (edge_thickness + (spine_thickness - edge_thickness) * grind)
    n_outline = len(outline)
    verts = np.concatenate(
        [
            np.column_stack([outline[:, 0], +half_thickness, outline[:, 1]]),
            np.column_stack([outline[:, 0], -half_thickness, outline[:, 1]]),
        ]
    )
    # Two faces of the outline, joined by a band of quads: a closed surface by construction.
    idx = np.arange(n_outline)
    nxt = (idx + 1) % n_outline
    fan = np.arange(1, n_outline - 1)
    faces = np.concatenate(
        [
            np.column_stack([idx, nxt, nxt + n_outline]),
            np.column_stack([idx, nxt + n_outline, idx + n_outline]),
            np.column_stack([np.zeros_like(fan), fan, fan + 1]),
            np.column_stack([np.zeros_like(fan), fan + 1, fan]) + n_outline,
        ]
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.fix_normals()
    return mesh


def cached_obj(mesh, name):
    """Return the path of 'mesh' on disk, exporting it once.

    'gs.morphs.Mesh' reads from file, and the particle sampler and signed distance field it reaches are
    both cached by file, so a mesh built in memory is written out rather than rebuilt on every run.
    """
    path = Path(get_cache_dir()) / "meat_cut" / f"{name}.obj"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(path)
        gs.logger.info(f"Wrote ~~<{path}>~~: {len(mesh.faces)} faces, extents {np.round(mesh.extents, 4)}.")
    return str(path)


def knife_rest_pose(entity):
    """Return the base-link quaternion and translation that stand a knife on its edge, blade along +x.

    The two knife entities carry rest transforms of their own -- the USD parser bakes the asset's stage
    transform into the entity, while the exported wedge has none -- and the parser splits that transform
    between the pose and the vertices, so no single stored quaternion describes it. Both are therefore
    recovered from the geometry: the rotation that carries the asset vertices onto their world positions
    is fitted, and its inverse is the base-link quaternion, which stands the knife up because the asset
    frame has the sharp edge on -z. What remains is a pure translation from the base link to the asset
    origin, so 'pos + quat * (asset_point + translation)' places any asset point in the world.
    """
    link = entity.links[0]
    geom = link.geoms[0] if link.geoms else link.vgeoms[0]
    init = geom.init_verts if link.geoms else geom.init_vverts

    def world_verts():
        return tensor_to_array(geom.get_verts() if link.geoms else geom.get_vverts())

    entity.set_qpos(np.concatenate([gu.zero_pos(), gu.identity_quat()]))
    posed = world_verts()
    u, _, vt = np.linalg.svd((init - init.mean(axis=0)).T @ (posed - posed.mean(axis=0)))
    quat = gu.R_to_quat(u @ vt)

    entity.set_qpos(np.concatenate([gu.zero_pos(), quat]))
    posed = world_verts()
    if not np.allclose(np.ptp(posed, axis=0), np.ptp(init, axis=0), atol=1e-4):
        gs.raise_exception(f"Knife did not stand up: extents {np.ptp(posed, axis=0)} against {np.ptp(init, axis=0)}.")
    return quat, posed.mean(axis=0) - init.mean(axis=0)


def active_meat_particles(meat):
    """Return the positions and local indices of active meat particles."""
    active = tensor_to_array(meat.get_particles_active()).astype(bool, copy=False)
    indices = np.flatnonzero(active)
    return tensor_to_array(meat.get_particles_pos())[indices], indices


def prune_meat_fragments(meat, particle_size):
    """Permanently deactivate tiny fragments and extremely sparse strands."""
    particles, active_indices = active_meat_particles(meat)
    if len(particles) == 0:
        return

    # This radius connects face- and edge-adjacent samples on the original regular lattice. A healthy
    # boundary particle still has several neighbours, whereas spray and a one-particle-wide filament do not.
    tree = KDTree(particles)
    radius = 1.75 * particle_size
    graph = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
    _, labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(labels)
    neighbour_counts = tree.query_ball_point(particles, radius, return_length=True)
    remove = (component_sizes[labels] < MIN_FRAGMENT_PARTICLES) | (neighbour_counts < MIN_LOCAL_NEIGHBORS)
    if not remove.any():
        return

    removed_indices = active_indices[remove]
    meat.set_particles_active(False, particles_idx_local=removed_indices)
    gs.logger.info(f"Removed ~~<{len(removed_indices)}>~~ sparse meat particles.")


def report_cut(meat, particle_size, when):
    """Log how deep the knife's cut runs and whether the slab has come apart.

    Two things are worth knowing about a cut and they are not the same. Whether the meat is in one piece
    or two is read from the particles directly: two particles belong to the same piece when they sit
    within a neighbourhood of each other, so labelling that graph counts the pieces. How far the cut runs
    is separate, because a slab still joined by a thin bridge at the board is the interesting case -- the
    kerf is found per height as the widest empty band across the cut plane, and the cut reaches as deep as
    the lowest height whose band is still open.
    """
    particles, _ = active_meat_particles(meat)
    # The sampler lays particles on a lattice of pitch 'particle_size', so a neighbourhood just above that
    # pitch keeps intact material in one piece while still registering a kerf as a break. A wider radius
    # bridges the cut and reports the slab as whole; a narrower one shatters the lattice under any strain.
    tree = KDTree(particles)
    graph = tree.sparse_distance_matrix(tree, 1.25 * particle_size, output_type="coo_matrix")
    n_pieces, labels = connected_components(graph, directed=False)
    sizes = np.sort(np.bincount(labels))[::-1]

    z_lo, z_hi = particles[:, 2].min(), particles[:, 2].max()
    open_depth, widest = z_hi, 0.0
    for z in np.arange(z_lo, z_hi, particle_size):
        layer = particles[(particles[:, 2] >= z) & (particles[:, 2] < z + particle_size)]
        if len(layer) < 2:
            continue
        gap = np.diff(np.sort(layer[:, 1])).max()
        # A band only counts as cut once it is clearly wider than the lattice pitch, which every layer
        # shows between neighbouring particles.
        if gap > 2.0 * particle_size:
            open_depth, widest = min(open_depth, z), max(widest, gap)
    cut = (z_hi - open_depth) / (z_hi - z_lo) if z_hi > z_lo else 0.0
    verdict = "severed" if n_pieces > 1 and sizes[1] > 0.05 * sizes[0] else "joined"
    gs.logger.info(
        f"[{when}] {verdict}: {n_pieces} piece(s) {sizes[:2].tolist()}, cut {100.0 * cut:.0f}% deep, "
        f"kerf {1000.0 * widest:.1f} mm, height {1000.0 * (z_hi - z_lo):.1f} mm"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--grid-density", type=float, default=224.0, help="MPM cells per meter")
    parser.add_argument("-s", "--substeps", type=int, default=40, help="MPM substeps per step")
    # The knife's blade is only as wide as it is: a slab taller than that can be scored but never cut
    # through, whatever the physics does, so the asset is scaled to a steak the blade can pass through.
    parser.add_argument("--meat-scale", type=float, default=0.40, help="Scale applied to the meat asset")
    parser.add_argument("--spine", type=float, default=0.012, help="Collision blade thickness at the spine")
    parser.add_argument("-E", "--young", type=float, default=4.0e4, help="Meat stiffness")
    parser.add_argument("-y", "--yield-stress", type=float, default=5.0e3, help="Stress past which meat stays deformed")
    parser.add_argument("--blade-rho", type=float, default=2.0e5, help="Density standing in for the grip on the knife")
    parser.add_argument("--no-cpic", action="store_true", help="Disable CPIC, so the knife dents instead of cutting")
    parser.add_argument(
        "--chop",
        type=float,
        default=0.0,
        help="Run a scripted stroke instead of reading keys, cutting this fraction of the meat height",
    )
    parser.add_argument("--headless", action="store_true", help="Run without the interactive viewer")
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps (0 runs until quit)")
    args = parser.parse_args()

    gs.init(backend=gs.cuda, precision="32", logging_level="info")

    meat_asset = meat_mesh(args.meat_scale)
    knife_asset = load_usd_mesh(KNIFE_USDZ)
    blade_asset = blade_mesh(0.002, args.spine)
    meat_file = cached_obj(meat_asset, f"meat_z90_{round(1000 * args.meat_scale)}")
    blade_file = cached_obj(blade_asset, f"blade_{round(1e4 * args.spine)}")
    meat_extents = meat_asset.extents
    particle_size = 0.01 * 64.0 / args.grid_density
    blade_width = blade_asset.extents[2]
    if meat_extents[2] > blade_width:
        gs.logger.warning(
            f"Meat is {1000 * meat_extents[2]:.0f} mm tall but the blade is only {1000 * blade_width:.0f} mm wide, "
            "so the knife cannot reach through it. Lower '--meat-scale' to sever the slab."
        )
    board_size = (1.7 * meat_extents[0], 2.4 * meat_extents[1], 0.02)
    # The domain only has to hold the meat plus the room its pieces need once severed; every cell beyond
    # that is swept for nothing. The solver then keeps a three-cell safety padding on each side, which is
    # added back here so the meat starts inside the usable region.
    room = np.array([0.04, 0.03, 0.0]) + 3.5 / args.grid_density
    domain_lower = np.array([-0.5 * meat_extents[0], -0.5 * meat_extents[1], 0.0]) - room
    domain_upper = np.array([0.5 * meat_extents[0], 0.5 * meat_extents[1], 2.0 * meat_extents[2]]) + room

    # Frame the meat rather than the domain: the knife sweeps well above it, and a view tied to the slab
    # keeps the cut readable at any '--meat-scale'.
    focus = (0.0, 0.0, 0.4 * meat_extents[2])
    eye = (1.15 * meat_extents[0], -1.5 * meat_extents[1], 1.6 * meat_extents[2])

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=args.substeps,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=domain_lower,
            upper_bound=domain_upper,
            grid_density=args.grid_density,
            enable_CPIC=not args.no_cpic,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=eye,
            camera_lookat=focus,
            camera_fov=45,
        ),
        show_viewer=not args.headless,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(collision=False),
        material=gs.materials.Rigid(needs_coup=False),
        surface=gs.surfaces.Rough(color=(0.22, 0.23, 0.26, 1.0)),
    )
    # The MPM domain floor is frictionless, so the meat needs a real surface underneath to be held down
    # while the knife drags across it.
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, -0.5 * board_size[2]),
            size=board_size,
            fixed=True,
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.6,
        ),
        surface=gs.surfaces.Rough(color=(0.68, 0.50, 0.31, 1.0)),
    )
    meat = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=meat_file,
            pos=(0.0, 0.0, 0.5 * meat_extents[2]),
        ),
        material=gs.materials.MPM.ElastoPlastic(
            E=args.young,
            nu=0.35,
            rho=1050.0,
            von_mises_yield_stress=args.yield_stress,
            sampler="regular",
        ),
        surface=gs.surfaces.Rough(
            color=(0.78, 0.30, 0.33, 1.0),
            roughness=0.72,
            vis_mode="recon",
            recon_backend="splashsurf",
        ),
    )
    knife_visual = scene.add_entity(
        morph=gs.morphs.USD(
            file=str(KNIFE_USDZ),
            fixed=False,
            collision=False,
            convexify=False,
            decimate=False,
        ),
        material=gs.materials.Rigid(needs_coup=False),
    )
    knife = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=blade_file,
            fixed=False,
            visualization=False,
            convexify=False,
            decimate=False,
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            # The blade is a thin wedge, so at any plausible density it weighs a fraction of the meat and
            # the reaction from cutting throws it off its commanded path between two steps, however often
            # the pose is rewritten. Held in a hand it would not budge: the mass stands in for that grip.
            rho=args.blade_rho,
            gravity_compensation=1.0,
            coup_friction=0.05,
            coup_softness=0.004,
            sdf_cell_size=0.0006,
            sdf_max_res=192,
        ),
    )

    scene.build(n_envs=0)

    knife_quat, knife_origin = knife_rest_pose(knife)
    visual_quat, visual_origin = knife_rest_pose(knife_visual)
    # Everything is commanded through one point on the knife: the middle of the cutting edge. Driving the
    # edge rather than a link origin means the requested height directly decides whether the meat is scored
    # or severed.
    anchor = np.array([blade_asset.bounds[:, 0].mean(), 0.0, blade_asset.bounds[0, 2]], dtype=gs.np_float)

    home_pos = np.array([0.0, 0.0, 1.3 * meat_extents[2]], dtype=gs.np_float)
    knife_pos = home_pos.copy()
    move_cmd = np.zeros(3, dtype=gs.np_float)

    def move(axis, direction):
        move_cmd[axis] += direction

    if not args.headless:
        scene.viewer.register_keybinds(
            Keybind("knife_back", Key.UP, KeyAction.HOLD, callback=move, args=(0, 1.0)),
            Keybind("knife_forward", Key.DOWN, KeyAction.HOLD, callback=move, args=(0, -1.0)),
            Keybind("knife_left", Key.LEFT, KeyAction.HOLD, callback=move, args=(1, 1.0)),
            Keybind("knife_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=(1, -1.0)),
            Keybind("knife_up", Key.E, KeyAction.HOLD, callback=move, args=(2, 1.0)),
            Keybind("knife_down", Key.Q, KeyAction.HOLD, callback=move, args=(2, -1.0)),
        )

    dt = scene.sim_options.dt
    # The scripted stroke settles, sinks the edge to the requested depth, draws the blade along its own
    # length the way a slicing stroke does, pushes sideways to open what it cut, then lifts clear.
    descent = home_pos[2] - (1.0 - args.chop) * meat_extents[2]
    chop_phases = np.cumsum([0.4, descent / 0.10, 0.8, 0.5])
    step = 0
    while args.steps == 0 or step < args.steps:
        if step > 0 and step % FRAGMENT_CLEANUP_INTERVAL == 0:
            prune_meat_fragments(meat, particle_size)

        if args.chop > 0.0:
            sim_time = step * dt
            if chop_phases[0] < sim_time <= chop_phases[1]:
                move(2, -1.0)
            elif chop_phases[1] < sim_time <= chop_phases[2]:
                if sim_time - dt <= chop_phases[1]:
                    report_cut(meat, particle_size, "at depth")
                move(0, 1.0)
            elif chop_phases[2] < sim_time <= chop_phases[3]:
                move(1, 1.0)
            elif sim_time > chop_phases[3]:
                if sim_time - dt <= chop_phases[3]:
                    report_cut(meat, particle_size, "pushed")
                move(2, 1.0)

        update_visualizer = not args.headless and step % VISUAL_UPDATE_STRIDE == 0
        lin_vel = 0.10 * move_cmd
        ang_vel = np.zeros(3, dtype=gs.np_float)
        # Scripted commands are generated on every physics step. Interactive HOLD callbacks, however, are
        # evaluated by the viewer only when it updates; preserve their most recent value between visual
        # updates so decimating SplashSurf reconstruction does not also slow the knife down.
        if args.chop > 0.0 or update_visualizer:
            move_cmd[:] = 0.0

        knife_pos += lin_vel * dt
        # The edge must stay above the board, or it pushes meat through a surface it cannot leave.
        knife_pos[2] = max(knife_pos[2], 0.0)

        for entity, rest_quat, origin in (
            (knife, knife_quat, knife_origin),
            (knife_visual, visual_quat, visual_origin),
        ):
            entity.set_qpos(
                np.concatenate([knife_pos - anchor - origin, rest_quat]),
            )
            entity.set_dofs_velocity(np.concatenate([lin_vel, ang_vel]))

        scene.step(update_visualizer=update_visualizer)
        step += 1

        if not args.headless and not scene.viewer.is_alive():
            break

    if "PYTEST_VERSION" not in os.environ:
        gs.logger.info(f"Ran ~~<{step}>~~ steps with ~~<{meat.n_particles}>~~ meat particles.")
        report_cut(meat, particle_size, "blade out")


if __name__ == "__main__":
    main()
