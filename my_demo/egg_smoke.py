"""A small whole-egg-inspired liquid-to-gel demo in a frying pan.

This is an intentionally simple, time-driven liquid-to-gel demo. MPM particles
begin with a viscous liquid constitutive model and are gradually switched to a
soft, nearly incompressible elastic model while their reconstructed surface
changes from nearly transparent white to opaque white, yellow, and finally black.
A soft orange yolk falls with the white and remains visually distinct.
The original large demo egg is retained inside a proportionally enlarged pan.
After cooking, the white continues from yellow to a charred black.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import quadrants as qd
import trimesh

import genesis as gs
import genesis.utils.geom as gu


SIM_DT = 1.0 / 60.0
SIM_SUBSTEPS = 64
GRID_DENSITY = 64
DEFAULT_DURATION = 10.0
CURE_START_TIME = 1.2
CURE_END_TIME = 5.5
WHITE_TIME = 1.0
YELLOW_TIME = 2.0
BURN_END_TIME = 3.5
RANDOM_SEED = 42

# MPM liquid has viscosity but no explicit surface-tension term. This force is
# evaluated at every solver substep, keeping the drop cohesive during impact.
COHESION_RADIUS = 0.13
COHESION_STRENGTH = 300.0
HORIZONTAL_DAMPING = 10.0

# Raw albumen is almost colorless. It first turns milky white as proteins
# coagulate, then develops only a subtle warm tint near the end of cooking.
# A little transmittance remains so the embedded yolk is not depth-occluded.
START_COLOR = np.array((0.97, 0.99, 1.00, 0.12), dtype=np.float32)
WHITE_COLOR = np.array((1.00, 1.00, 1.00, 0.86), dtype=np.float32)
YELLOW_COLOR = np.array((1.00, 0.86, 0.34, 0.94), dtype=np.float32)
BURNT_COLOR = np.array((0.24, 0.075, 0.018, 1.00), dtype=np.float32)
START_ROUGHNESS = 0.03
END_ROUGHNESS = 0.88

YOLK_COLOR = (1.00, 0.43, 0.035, 1.00)
YOLK_RADIUS = 0.055
YOLK_VERTICAL_SCALE = 0.78
SCENE_LIFT = 0.12
ALBUMEN_POSITION = (0.0, 0.0, 0.16 + SCENE_LIFT)
ALBUMEN_RADIUS = 0.12
YOLK_POSITION = (0.0, 0.0, 0.23 + SCENE_LIFT)
YOLK_CAVITY_RADIUS = 0.063

# Keep the egg at its original demo scale and enlarge the pan around it. The
# scene is now intentionally display-scaled rather than strictly life-sized.
PAN_USDZ = Path(__file__).with_name("frying_pan_3d_model.usdz")
PAN_SCALE = 0.38
DEFAULT_RECORD_PATH = Path.home() / "egg_demo.mp4"
PAN_MOTION_START = 0.35
PAN_MOTION_RAMP = 0.65
PAN_ORBIT_RADIUS = 0.05
PAN_ORBIT_PERIOD = 1.4
PAN_YAW_AMPLITUDE = math.radians(16.0)
PAN_LIFT_AMPLITUDE = 0.035
PAN_COLLISION_ENABLED = True
PAN_POSITION_GAIN = 20.0
PAN_ORIENTATION_GAIN = 16.0
PAN_COUP_FRICTION = 0.12
PAN_COLLIDER_FLAT_RADIUS = 0.270
PAN_COLLIDER_INNER_RADIUS = 0.325
PAN_COLLIDER_OUTER_RADIUS = 0.352
PAN_COLLIDER_FLOOR_Z = 0.023
PAN_COLLIDER_RIM_Z = 0.132
PAN_COLLIDER_UNDERSIDE_Z = 0.0
PAN_COLLIDER_SEGMENTS = 64

# The rasterizer does not expose Genesis' Stable Fluids density grid as a
# renderable volume.  Use soft, camera-facing wisps instead: unlike sphere
# particles these have feathered, irregular silhouettes and cannot look like
# soap bubbles.  They are kinematic and cannot push the pan or egg.
SMOKE_START = 0.55
SMOKE_RAMP = 2.0
SMOKE_INTERVAL = 0.22
SMOKE_POOL_SIZE = 32
SMOKE_SOURCE_INNER_RADIUS = 0.02
SMOKE_SOURCE_OUTER_RADIUS = 0.17
SMOKE_SOURCE_HEIGHT = 0.085
SMOKE_WISP_WIDTH = 0.013
SMOKE_WISP_HEIGHT = 0.17
SMOKE_MIN_LIFETIME = 2.0
SMOKE_MAX_LIFETIME = 3.0
SMOKE_MIN_SPEED = 0.065
SMOKE_MAX_SPEED = 0.13
SMOKE_COLOR = (0.96, 0.98, 1.0)
SMOKE_HIDDEN_POS = (0.0, 0.0, -2.0)

CAMERA_RES = (1440, 1080)
CAMERA_POS = (1.45, -1.80, 1.20 + SCENE_LIFT)
CAMERA_LOOKAT = (0.0, 0.10, 0.16 + SCENE_LIFT)
CAMERA_FOV = 36


def smoothstep(value: float) -> float:
    """Clamp ``value`` to [0, 1] and apply a cubic ease-in/ease-out."""
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def cure_progress(sim_time: float) -> float:
    """Return the scheduled liquid-to-gel progress at ``sim_time``."""
    normalized_time = (sim_time - CURE_START_TIME) / (CURE_END_TIME - CURE_START_TIME)
    return smoothstep(normalized_time)


def color_progress(sim_time: float) -> float:
    """Return progress over the complete raw-to-burnt appearance timeline."""
    return smoothstep(sim_time / BURN_END_TIME)


def interpolate_surface_color(sim_time: float) -> np.ndarray:
    """Interpolate raw -> white -> yellow -> charred black by simulation time."""
    if sim_time < WHITE_TIME:
        progress = smoothstep(sim_time / WHITE_TIME)
        return START_COLOR + (WHITE_COLOR - START_COLOR) * progress
    if sim_time < YELLOW_TIME:
        progress = smoothstep((sim_time - WHITE_TIME) / (YELLOW_TIME - WHITE_TIME))
        return WHITE_COLOR + (YELLOW_COLOR - WHITE_COLOR) * progress
    progress = smoothstep((sim_time - YELLOW_TIME) / (BURN_END_TIME - YELLOW_TIME))
    return YELLOW_COLOR + (BURNT_COLOR - YELLOW_COLOR) * progress


def interpolate_roughness(progress: float) -> float:
    """Make the glossy liquid grow matte as it cures into a gel."""
    progress = smoothstep(progress)
    return START_ROUGHNESS + (END_ROUGHNESS - START_ROUGHNESS) * progress


def create_albumen_shell_mesh() -> trimesh.Trimesh:
    """Create a closed albumen volume with a yolk-sized cavity near its top."""
    outer = trimesh.creation.icosphere(subdivisions=3, radius=ALBUMEN_RADIUS)
    inner = trimesh.creation.icosphere(subdivisions=3, radius=YOLK_CAVITY_RADIUS)
    cavity_offset = np.asarray(YOLK_POSITION) - np.asarray(ALBUMEN_POSITION)
    inner.apply_translation(cavity_offset)

    # Reversing the inner winding makes signed-distance sampling treat it as
    # empty space instead of filling it with a second set of white particles.
    inner.faces = inner.faces[:, ::-1]
    return trimesh.util.concatenate((outer, inner))


def create_yolk_mesh() -> trimesh.Trimesh:
    """Create a slightly flattened yolk dome instead of a perfect sphere."""
    yolk = trimesh.creation.icosphere(subdivisions=3, radius=YOLK_RADIUS)
    yolk.apply_scale((1.0, 1.0, YOLK_VERTICAL_SCALE))
    return yolk


def create_pan_collision_mesh() -> trimesh.Trimesh:
    """Create a closed, thick, concave bowl used only for MPM collision."""
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, PAN_COLLIDER_FLOOR_Z)]

    def append_ring(radius: float, z: float) -> int:
        start = len(vertices)
        for segment in range(PAN_COLLIDER_SEGMENTS):
            angle = 2.0 * math.pi * segment / PAN_COLLIDER_SEGMENTS
            vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
        return start

    flat_ring = append_ring(PAN_COLLIDER_FLAT_RADIUS, PAN_COLLIDER_FLOOR_Z)
    inner_rim = append_ring(PAN_COLLIDER_INNER_RADIUS, PAN_COLLIDER_RIM_Z)
    outer_rim = append_ring(PAN_COLLIDER_OUTER_RADIUS, PAN_COLLIDER_RIM_Z)
    outer_bottom = append_ring(PAN_COLLIDER_OUTER_RADIUS, PAN_COLLIDER_UNDERSIDE_Z)
    underside_center = len(vertices)
    vertices.append((0.0, 0.0, PAN_COLLIDER_UNDERSIDE_Z))

    faces: list[tuple[int, int, int]] = []
    for segment in range(PAN_COLLIDER_SEGMENTS):
        next_segment = (segment + 1) % PAN_COLLIDER_SEGMENTS

        # Upward-facing cooking floor.
        faces.append((0, flat_ring + segment, flat_ring + next_segment))

        # Sloped inner wall: its outward normal points into the bowl cavity.
        faces.append((flat_ring + segment, inner_rim + segment, inner_rim + next_segment))
        faces.append((flat_ring + segment, inner_rim + next_segment, flat_ring + next_segment))

        # Top rim, exterior wall, and downward-facing underside close the solid.
        faces.append((inner_rim + segment, outer_rim + segment, outer_rim + next_segment))
        faces.append((inner_rim + segment, outer_rim + next_segment, inner_rim + next_segment))
        faces.append((outer_rim + segment, outer_bottom + segment, outer_bottom + next_segment))
        faces.append((outer_rim + segment, outer_bottom + next_segment, outer_rim + next_segment))
        faces.append((underside_center, outer_bottom + next_segment, outer_bottom + segment))

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.fix_normals()
    if not mesh.is_watertight:
        raise RuntimeError("Generated frying-pan collision proxy is not watertight")
    return mesh


def create_smoke_wisp_mesh(width: float, height: float, variant: int) -> trimesh.Trimesh:
    """Create a narrow, curling steam ribbon with feathered vertex alpha."""
    normal = np.asarray(CAMERA_POS, dtype=np.float32) - np.asarray(CAMERA_LOOKAT, dtype=np.float32)
    normal /= np.linalg.norm(normal)
    right = np.cross(np.array((0.0, 0.0, 1.0), dtype=np.float32), normal)
    right /= np.linalg.norm(right)
    up = np.cross(normal, right)
    rows = 15
    columns = 5
    rgb = tuple(np.round(255.0 * np.asarray(SMOKE_COLOR)).astype(np.uint8))
    phase = 0.71 * variant
    vertices: list[np.ndarray] = []
    colors: list[tuple[int, int, int, int]] = []
    faces: list[tuple[int, int, int]] = []

    for row in range(rows):
        y = row / (rows - 1)
        center_offset = width * (
            0.55 * math.sin(1.65 * math.pi * y + phase) + 0.20 * math.sin(4.8 * math.pi * y - 0.43 * phase)
        )
        local_half_width = 0.5 * width * (0.38 + 0.76 * y)
        density = 0.58 + 0.22 * math.sin(5.1 * math.pi * y + 1.3 * phase)
        vertical_fade = math.sin(math.pi * y) ** 0.52
        for column in range(columns):
            across = -1.0 + 2.0 * column / (columns - 1)
            pos = (center_offset + local_half_width * across) * right + height * y * up
            edge_fade = max(0.0, 1.0 - across * across) ** 1.5
            alpha = round(104.0 * density * vertical_fade * edge_fade)
            vertices.append(pos.astype(np.float32))
            colors.append((*rgb, alpha))

    for row in range(rows - 1):
        for column in range(columns - 1):
            lower_left = row * columns + column
            lower_right = lower_left + 1
            upper_left = lower_left + columns
            upper_right = upper_left + 1
            faces.append((lower_left, lower_right, upper_right))
            faces.append((lower_left, upper_right, upper_left))

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.asarray(colors, dtype=np.uint8),
    )
    return mesh


def pan_motion(sim_time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return target offset, physical velocities, and target yaw."""
    ramp_input = (sim_time - PAN_MOTION_START) / PAN_MOTION_RAMP
    ramp = smoothstep(ramp_input)
    ramp_rate = 0.0
    if 0.0 < ramp_input < 1.0:
        ramp_rate = 6.0 * ramp_input * (1.0 - ramp_input) / PAN_MOTION_RAMP

    phase_rate = 2.0 * math.pi / PAN_ORBIT_PERIOD
    phase = 2.0 * math.pi * sim_time / PAN_ORBIT_PERIOD
    pos = np.array(
        (
            PAN_ORBIT_RADIUS * ramp * math.cos(phase),
            PAN_ORBIT_RADIUS * ramp * math.sin(phase),
            0.5 * PAN_LIFT_AMPLITUDE * ramp * (1.0 - math.cos(phase)),
        ),
        dtype=np.float32,
    )
    vel = np.array(
        (
            PAN_ORBIT_RADIUS * (ramp_rate * math.cos(phase) - ramp * phase_rate * math.sin(phase)),
            PAN_ORBIT_RADIUS * (ramp_rate * math.sin(phase) + ramp * phase_rate * math.cos(phase)),
            0.5 * PAN_LIFT_AMPLITUDE * (ramp_rate * (1.0 - math.cos(phase)) + ramp * phase_rate * math.sin(phase)),
        ),
        dtype=np.float32,
    )
    yaw_rate = PAN_YAW_AMPLITUDE * (ramp_rate * math.sin(phase) + ramp * phase_rate * math.cos(phase))
    angular_vel = np.array((0.0, 0.0, yaw_rate), dtype=np.float32)
    yaw = PAN_YAW_AMPLITUDE * ramp * math.sin(phase)
    return pos, vel, angular_vel, yaw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop a translucent egg white and soft yolk, then gradually cook the white."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help=f"simulation duration in seconds (default: {DEFAULT_DURATION:g})",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="disable the realtime viewer (useful for smoke tests)",
    )
    parser.add_argument(
        "--backend",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Genesis compute backend (default: cuda)",
    )
    recording_group = parser.add_mutually_exclusive_group()
    recording_group.add_argument(
        "--record",
        type=Path,
        default=DEFAULT_RECORD_PATH,
        metavar="OUTPUT.mp4",
        help=f"record to a custom MP4 path (default: {DEFAULT_RECORD_PATH})",
    )
    recording_group.add_argument(
        "--no-record",
        action="store_const",
        const=None,
        dest="record",
        help="disable the default MP4 recording",
    )
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0.0:
        parser.error("--duration must be a positive finite number")
    if args.record is not None and args.record.suffix.lower() != ".mp4":
        parser.error("--record output filename must end in .mp4")
    return args


def update_surface(surface: gs.surfaces.Surface, sim_time: float) -> None:
    """Update liquid color, transparency, and gloss for surface reconstruction."""
    rgba = interpolate_surface_color(sim_time)
    roughness = interpolate_roughness(color_progress(sim_time))
    surface.update_texture(
        color_texture=gs.textures.ColorTexture(color=tuple(float(channel) for channel in rgba[:3])),
        opacity_texture=gs.textures.ColorTexture(color=(float(rgba[3]),)),
        roughness_texture=gs.textures.ColorTexture(color=(roughness,)),
        force=True,
    )


@qd.func
def egg_white_cohesion(pos, vel, _time, _particle_idx):
    """Horizontal surface-tension proxy evaluated at every MPM substep."""
    offset_xy = qd.Vector([pos[0], pos[1], 0.0], dt=gs.qd_float)
    radius = offset_xy.norm(1.0e-6)
    acceleration = qd.Vector(
        [-HORIZONTAL_DAMPING * vel[0], -HORIZONTAL_DAMPING * vel[1], 0.0],
        dt=gs.qd_float,
    )
    if radius > COHESION_RADIUS:
        acceleration -= COHESION_STRENGTH * (radius - COHESION_RADIUS) * offset_xy / radius
    return acceleration


def main() -> None:
    args = parse_args()

    record_path = None
    if args.record is not None:
        record_path = args.record.expanduser().resolve()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        # VideoEncoder opens its container lazily on the first background-
        # encoded frame. Create the destination now so it is immediately
        # visible in Finder from the moment main() starts. It becomes playable
        # after stop_recording() writes the final MP4 index.
        record_path.touch(exist_ok=True)
        print(f"[egg demo] Recording from frame 0 to: {record_path}", flush=True)
        print("[egg demo] The MP4 becomes playable when the simulation exits.", flush=True)

    backend = gs.cuda if args.backend == "cuda" else gs.cpu
    gs.init(backend=backend, precision="32", logging_level="info")

    if not PAN_USDZ.is_file():
        raise FileNotFoundError(f"Frying-pan asset not found: {PAN_USDZ}")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=SIM_DT,
            substeps=SIM_SUBSTEPS,
            gravity=(0.0, 0.0, -9.81),
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        mpm_options=gs.options.MPMOptions(
            # The boundary is only a safety net below the pan. Rigid–MPM
            # coupling with the pan mesh provides the actual cooking surface.
            lower_bound=(-0.6, -0.6, -3.0 / GRID_DENSITY),
            upper_bound=(0.6, 0.6, 0.9),
            grid_density=GRID_DENSITY,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=CAMERA_RES,
            refresh_rate=60,
            realtime_factor=1.0,
            camera_pos=CAMERA_POS,
            camera_lookat=CAMERA_LOOKAT,
            camera_fov=CAMERA_FOV,
        ),
        vis_options=gs.options.VisOptions(
            background_color=(0.055, 0.065, 0.085),
            ambient_light=(0.22, 0.22, 0.24),
            plane_reflection=True,
            lights=[
                {
                    "type": "directional",
                    "dir": (-1.0, 1.0, -2.0),
                    "color": (1.0, 0.96, 0.90),
                    "intensity": 4.0,
                },
                {
                    "type": "directional",
                    "dir": (1.0, -0.5, -1.0),
                    "color": (0.75, 0.85, 1.0),
                    "intensity": 1.5,
                },
            ],
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=not args.headless,
    )

    # Unlike the earlier once-per-frame correction, a force field participates
    # in every one of the 64 MPM substeps and can oppose impact spreading before
    # particles have already escaped from the main puddle.
    scene.add_force_field(gs.force_fields.Custom(egg_white_cohesion))

    # The countertop is visual only; the pan below supplies MPM collision.
    scene.add_entity(
        morph=gs.morphs.Plane(pos=(0.0, 0.0, -0.003), collision=False),
        material=gs.materials.Rigid(needs_coup=False),
        surface=gs.surfaces.Rough(color=(0.33, 0.35, 0.39, 1.0)),
    )

    frying_pan_visual = scene.add_entity(
        morph=gs.morphs.USD(
            file=str(PAN_USDZ),
            prim_path="/frying_pan_3d_model",
            pos=(0.0, 0.0, SCENE_LIFT),
            scale=PAN_SCALE,
            fixed=False,
            collision=False,
            convexify=False,
            decimate=False,
            watertighten=0,
        ),
        material=gs.materials.Rigid(needs_coup=False, gravity_compensation=1.0),
    )

    # The downloaded USD consists of open, paper-thin surfaces and cannot
    # provide a reliable signed distance field. This invisible closed bowl is
    # the physical pan; the USD above remains the visible shell.
    frying_pan = scene.add_entity(
        morph=gs.morphs.MeshSet(
            files=(create_pan_collision_mesh(),),
            poss=((0.0, 0.0, 0.0),),
            eulers=((0.0, 0.0, 0.0),),
            pos=(0.0, 0.0, SCENE_LIFT),
            fixed=False,
            visualization=False,
            collision=True,
            convexify=False,
            decimate=False,
            watertighten=0,
            align=False,
        ),
        material=gs.materials.Rigid(
            needs_coup=PAN_COLLISION_ENABLED,
            coup_friction=PAN_COUP_FRICTION,
            # The pan is an actuated floating rigid body. Cancelling its own
            # weight prevents the commanded orbit from drifting downward;
            # contact impulses with the egg are still solved normally.
            gravity_compensation=1.0,
        ),
    )

    liquid_material = gs.materials.MPM.Liquid(
        rho=1030.0,
        viscous=True,
        E=5.0e4,
        nu=0.2,
        lam=8.0e4,
        mu=8.0e4,
        sampler="regular",
    )
    gel_material = gs.materials.MPM.Elastic(
        rho=1030.0,
        E=6.0e4,
        nu=0.45,
        sampler="regular",
        model="neohooken",
    )
    # A purely elastic yolk always recovers its original sphere, regardless of
    # how small E is.  Low-yield elasto-plasticity lets impact deformation
    # persist, closer to a membrane containing a thick liquid.
    yolk_material = gs.materials.MPM.ElastoPlastic(
        rho=1025.0,
        E=4.0e3,
        nu=0.45,
        sampler="regular",
        use_von_mises=True,
        von_mises_yield_stress=25.0,
    )

    # Register the target constitutive model before build.  The particles can
    # then change material_idx at runtime without rebuilding the scene.
    scene.mpm_solver.add_material(gel_material)

    egg_surface = gs.surfaces.Smooth(
        color=tuple(float(channel) for channel in START_COLOR),
        roughness=START_ROUGHNESS,
        vis_mode="recon",
        recon_backend="splashsurf",
    )
    egg_white = scene.add_entity(
        morph=gs.morphs.MeshSet(
            files=(create_albumen_shell_mesh(),),
            poss=(ALBUMEN_POSITION,),
            eulers=((0.0, 0.0, 0.0),),
        ),
        material=liquid_material,
        surface=egg_surface,
    )
    # The yolk starts inside the explicitly empty cavity in the white.  This
    # gives the appearance and motion of a single egg without overlapping two
    # solid particle volumes, which would double density and cause instability.
    egg_yolk = scene.add_entity(
        morph=gs.morphs.MeshSet(
            files=(create_yolk_mesh(),),
            poss=(YOLK_POSITION,),
            eulers=((0.0, 0.0, 0.0),),
        ),
        material=yolk_material,
        surface=gs.surfaces.Smooth(
            color=YOLK_COLOR,
            roughness=0.28,
            vis_mode="visual",
        ),
    )

    smoke_puffs = []
    for i in range(SMOKE_POOL_SIZE):
        width_scale = 0.78 + 0.38 * ((i % 5) / 4.0)
        height_scale = 0.82 + 0.30 * (((i * 3) % 7) / 6.0)
        smoke_surface = gs.surfaces.Smooth(
            # Surface alpha activates BLEND in the rasterizer; it is then
            # multiplied by the ribbon's per-vertex feathering.
            color=(*SMOKE_COLOR, 0.14),
            # A small emissive term keeps thin steam white in the pan's shadow.
            emissive=(0.15, 0.16, 0.17),
            roughness=1.0,
            smooth=False,
            double_sided=True,
        )
        smoke_puffs.append(
            scene.add_entity(
                morph=gs.morphs.MeshSet(
                    files=(
                        create_smoke_wisp_mesh(
                            SMOKE_WISP_WIDTH * width_scale,
                            SMOKE_WISP_HEIGHT * height_scale,
                            variant=i,
                        ),
                    ),
                    poss=((0.0, 0.0, 0.0),),
                    eulers=((0.0, 0.0, 0.0),),
                    pos=SMOKE_HIDDEN_POS,
                    convexify=False,
                    decimate=False,
                ),
                material=gs.materials.Kinematic(),
                surface=smoke_surface,
            )
        )

    record_camera = None
    if record_path is not None:
        record_camera = scene.add_camera(
            res=CAMERA_RES,
            pos=CAMERA_POS,
            lookat=CAMERA_LOOKAT,
            fov=CAMERA_FOV,
            GUI=False,
        )

    scene.build(n_envs=0)
    pan_initial_pos = gs.utils.tensor_to_array(frying_pan.get_pos()).astype(np.float32)
    pan_initial_quat = gs.utils.tensor_to_array(frying_pan.get_quat()).astype(np.float32)
    visual_pan_initial_pos = gs.utils.tensor_to_array(frying_pan_visual.get_pos()).astype(np.float32)
    visual_pan_initial_quat = gs.utils.tensor_to_array(frying_pan_visual.get_quat()).astype(np.float32)

    recording_started = False
    if record_camera is not None:
        record_camera.start_recording(save_to_filename=str(record_path), fps=60)
        recording_started = True
        # Camera recording normally begins after the first simulation step.
        # Pull its first deadline back to t=0 and explicitly capture the
        # untouched initial state, so the MP4 includes the whole fall.
        record_camera._recorded_t_next = scene._t
        record_camera.update_recording()
        gs.logger.info(f"Recording complete simulation to {record_path}.")

    rng = np.random.default_rng(RANDOM_SEED)
    smoke_rng = np.random.default_rng(RANDOM_SEED + 1)
    next_smoke_time = SMOKE_START
    smoke_birth_times = np.full(SMOKE_POOL_SIZE, -np.inf, dtype=np.float32)
    smoke_lifetimes = np.ones(SMOKE_POOL_SIZE, dtype=np.float32)
    smoke_origins = np.zeros((SMOKE_POOL_SIZE, 3), dtype=np.float32)
    smoke_velocities = np.zeros((SMOKE_POOL_SIZE, 3), dtype=np.float32)
    next_smoke_slot = 0
    cure_order = rng.permutation(egg_white.n_particles)
    material_idx_field = scene.mpm_solver.particles_info.material_idx
    material_indices = material_idx_field.to_numpy()
    cured_count = 0
    progress = 0.0

    total_steps = max(1, math.ceil(args.duration / SIM_DT))
    gs.logger.info(
        f"Whole-egg demo: {egg_white.n_particles} white + {egg_yolk.n_particles} yolk particles, "
        f"{total_steps} steps ({args.duration:.2f} s)."
    )

    try:
        for step in range(total_steps):
            sim_time = min((step + 1) * SIM_DT, args.duration)
            progress = cure_progress(sim_time)
            target_cured_count = min(egg_white.n_particles, round(progress * egg_white.n_particles))

            if target_cured_count > cured_count:
                newly_cured_local = cure_order[cured_count:target_cured_count]
                newly_cured_global = egg_white.particle_start + newly_cured_local
                material_indices[newly_cured_global] = gel_material.idx
                material_idx_field.from_numpy(material_indices)
                cured_count = target_cured_count

            # Drive a floating rigid pan through its six physical DOFs. The
            # rigid solver integrates this velocity and exposes it to the MPM
            # coupler, so any egg lift/drag comes from collision impulses.
            pan_pos_offset, motion_linear_vel, motion_angular_vel, pan_yaw = pan_motion(sim_time)
            pan_current_pos = gs.utils.tensor_to_array(frying_pan.get_pos()).astype(np.float32)
            pan_target_pos = pan_initial_pos + pan_pos_offset
            # A Cartesian velocity servo keeps the free rigid body centered on
            # its orbit despite the equal-and-opposite impulse from the egg.
            # It drives only the pan; MPM particle motion remains collision-only.
            pan_linear_vel = motion_linear_vel + PAN_POSITION_GAIN * (pan_target_pos - pan_current_pos)
            pan_current_quat = gs.utils.tensor_to_array(frying_pan.get_quat()).astype(np.float32)
            pan_yaw_quat = np.array(
                (math.cos(0.5 * pan_yaw), 0.0, 0.0, math.sin(0.5 * pan_yaw)),
                dtype=np.float32,
            )
            pan_target_quat = gu.transform_quat_by_quat(pan_initial_quat, pan_yaw_quat)
            pan_quat_error = gu.transform_quat_by_quat(gu.inv_quat(pan_current_quat), pan_target_quat)
            pan_angular_vel = motion_angular_vel + PAN_ORIENTATION_GAIN * gu.quat_to_rotvec(pan_quat_error)
            # Floating-base rotational DOFs are expressed in the body's local
            # frame, while the yaw trajectory and quaternion error above are
            # world-frame quantities.
            pan_angular_vel = gu.quat_to_R(pan_current_quat).T @ pan_angular_vel
            frying_pan.set_dofs_velocity(np.concatenate((pan_linear_vel, pan_angular_vel)).astype(np.float32))

            # Drive the visible USD through its own rigid root instead of
            # overwriting its pose. This preserves the asset's internal Y-up
            # conversion while making it follow the same world trajectory.
            visual_pan_current_pos = gs.utils.tensor_to_array(frying_pan_visual.get_pos()).astype(np.float32)
            visual_pan_linear_vel = motion_linear_vel + PAN_POSITION_GAIN * (
                visual_pan_initial_pos + pan_pos_offset - visual_pan_current_pos
            )
            visual_pan_current_quat = gs.utils.tensor_to_array(frying_pan_visual.get_quat()).astype(np.float32)
            visual_pan_target_quat = gu.transform_quat_by_quat(visual_pan_initial_quat, pan_yaw_quat)
            visual_pan_quat_error = gu.transform_quat_by_quat(
                gu.inv_quat(visual_pan_current_quat), visual_pan_target_quat
            )
            visual_pan_angular_vel = motion_angular_vel + PAN_ORIENTATION_GAIN * gu.quat_to_rotvec(
                visual_pan_quat_error
            )
            visual_pan_angular_vel = gu.quat_to_R(visual_pan_current_quat).T @ visual_pan_angular_vel
            frying_pan_visual.set_dofs_velocity(
                np.concatenate((visual_pan_linear_vel, visual_pan_angular_vel)).astype(np.float32)
            )

            # White kinematic puffs begin sparsely, then grow denser across the
            # whole cooking surface. They are visualization-only and therefore
            # cannot alter rigid--MPM collision or the pan trajectory.
            while sim_time >= next_smoke_time:
                smoke_progress = smoothstep((next_smoke_time - SMOKE_START) / SMOKE_RAMP)
                smoke_puff_count = 1 + int(smoke_progress >= 0.55)
                for _ in range(smoke_puff_count):
                    smoke_angle = smoke_rng.uniform(0.0, 2.0 * math.pi)
                    smoke_radius = math.sqrt(
                        smoke_rng.uniform(
                            SMOKE_SOURCE_INNER_RADIUS**2,
                            SMOKE_SOURCE_OUTER_RADIUS**2,
                        )
                    )
                    smoke_origins[next_smoke_slot] = pan_current_pos + np.array(
                        (
                            smoke_radius * math.cos(smoke_angle),
                            smoke_radius * math.sin(smoke_angle),
                            SMOKE_SOURCE_HEIGHT,
                        ),
                        dtype=np.float32,
                    )
                    drift_angle = smoke_rng.uniform(0.0, 2.0 * math.pi)
                    smoke_direction = np.array(
                        (
                            0.16 * math.cos(drift_angle),
                            0.16 * math.sin(drift_angle),
                            1.0,
                        ),
                        dtype=np.float32,
                    )
                    smoke_direction /= np.linalg.norm(smoke_direction)
                    smoke_velocities[next_smoke_slot] = smoke_direction * (
                        SMOKE_MIN_SPEED + (SMOKE_MAX_SPEED - SMOKE_MIN_SPEED) * smoke_progress
                    )
                    smoke_birth_times[next_smoke_slot] = next_smoke_time
                    smoke_lifetimes[next_smoke_slot] = smoke_rng.uniform(SMOKE_MIN_LIFETIME, SMOKE_MAX_LIFETIME)
                    next_smoke_slot = (next_smoke_slot + 1) % SMOKE_POOL_SIZE
                next_smoke_time += SMOKE_INTERVAL

            for smoke_idx, smoke_puff in enumerate(smoke_puffs):
                smoke_age = sim_time - smoke_birth_times[smoke_idx]
                if 0.0 <= smoke_age <= smoke_lifetimes[smoke_idx]:
                    smoke_pos = (
                        smoke_origins[smoke_idx]
                        + smoke_velocities[smoke_idx] * smoke_age
                        + np.array(
                            (
                                0.012 * math.sin(3.1 * smoke_age + smoke_idx),
                                0.010 * math.cos(2.7 * smoke_age + 0.7 * smoke_idx),
                                0.055 * smoke_age * smoke_age,
                            ),
                            dtype=np.float32,
                        )
                    )
                    smoke_puff.set_pos(smoke_pos, skip_forward=True)
                else:
                    smoke_puff.set_pos(SMOKE_HIDDEN_POS, skip_forward=True)

            # Primitive entities copy their input surface while constructing
            # the internal visual mesh. Update the entity-owned copy that the
            # rasterizer actually reads, not the original ``egg_surface``.
            update_surface(egg_white.surface, sim_time)
            scene.step(update_visualizer=not args.headless)
    except KeyboardInterrupt:
        gs.logger.info("Whole-egg demo interrupted by user.")
    finally:
        if recording_started:
            record_camera.stop_recording()
            if record_path.is_file():
                video_size_mb = record_path.stat().st_size / (1024.0 * 1024.0)
                gs.logger.info(f"Saved MP4 recording to {record_path} ({video_size_mb:.2f} MiB).")
            else:
                gs.logger.error(f"MP4 encoder closed but output file was not created: {record_path}")
        gs.logger.info(
            f"Whole-egg demo finished at white cure progress {progress:.1%}; "
            f"{cured_count}/{egg_white.n_particles} particles use the gel model."
        )


if __name__ == "__main__":
    main()
