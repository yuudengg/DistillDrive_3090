"""CARLA 0.9.15 smoke test: connect, load Town01, attach a 6-camera nuScenes-style
rig to one ego vehicle, run 200 ticks in synchronous mode, and record GPU memory /
per-tick timing. Does not modify any existing CARLA install files or configs.

Usage:
    conda activate carla015
    python tools/carla/smoke_test.py --port 3000
"""
import argparse
import json
import queue
import subprocess
import time

import carla


def gpu_mem_used_mb():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    return int(out.decode().strip().splitlines()[0])


# nuScenes-like 6-camera layout: (name, yaw_deg)
CAMERA_RIG = [
    ("CAM_FRONT", 0.0),
    ("CAM_FRONT_LEFT", -55.0),
    ("CAM_FRONT_RIGHT", 55.0),
    ("CAM_BACK", 180.0),
    ("CAM_BACK_LEFT", -110.0),
    ("CAM_BACK_RIGHT", 110.0),
]
IMG_W, IMG_H = 1600, 900


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--out", default="tools/carla/smoke_test_result.json")
    args = ap.parse_args()

    result = {}

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    server_version = client.get_server_version()
    client_version = client.get_client_version()
    result["server_version"] = server_version
    result["client_version"] = client_version
    print(f"[carla] server={server_version} client={client_version}")
    assert server_version.startswith("0.9.15"), f"expected 0.9.15, got {server_version}"

    result["gpu_mem_carla_only_mb"] = gpu_mem_used_mb()
    print(f"[gpu] CARLA only: {result['gpu_mem_carla_only_mb']} MB")

    world = client.load_world("Town01")
    time.sleep(2.0)  # let the map finish loading before measuring/spawning

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMG_W))
    cam_bp.set_attribute("image_size_y", str(IMG_H))

    cam_transforms = {
        "CAM_FRONT": carla.Transform(carla.Location(x=1.5, z=1.5), carla.Rotation(yaw=0)),
        "CAM_FRONT_LEFT": carla.Transform(carla.Location(x=1.0, y=-0.6, z=1.5), carla.Rotation(yaw=-55)),
        "CAM_FRONT_RIGHT": carla.Transform(carla.Location(x=1.0, y=0.6, z=1.5), carla.Rotation(yaw=55)),
        "CAM_BACK": carla.Transform(carla.Location(x=-1.5, z=1.5), carla.Rotation(yaw=180)),
        "CAM_BACK_LEFT": carla.Transform(carla.Location(x=-0.8, y=-0.6, z=1.5), carla.Rotation(yaw=-110)),
        "CAM_BACK_RIGHT": carla.Transform(carla.Location(x=-0.8, y=0.6, z=1.5), carla.Rotation(yaw=110)),
    }

    cameras = {}
    cam_queues = {}
    for name, _yaw in CAMERA_RIG:
        cam = world.spawn_actor(cam_bp, cam_transforms[name], attach_to=vehicle)
        q = queue.Queue()
        cam.listen(q.put)
        cameras[name] = cam
        cam_queues[name] = q

    vehicle.set_autopilot(False)

    # warmup: let rendering/cameras stabilize before measuring steady-state memory/timing
    for _ in range(10):
        world.tick()
        for q in cam_queues.values():
            q.get(timeout=10.0)

    result["gpu_mem_with_6cam_mb"] = gpu_mem_used_mb()
    print(f"[gpu] CARLA + 6 cameras (steady state): {result['gpu_mem_with_6cam_mb']} MB")

    tick_times_ms = []
    for i in range(args.ticks):
        t0 = time.perf_counter()
        world.tick()
        for name, q in cam_queues.items():
            img = q.get(timeout=10.0)
            assert img.width == IMG_W and img.height == IMG_H
        t1 = time.perf_counter()
        tick_times_ms.append((t1 - t0) * 1000.0)

    tick_times_ms.sort()
    n = len(tick_times_ms)
    result["ticks"] = n
    result["tick_ms_mean"] = sum(tick_times_ms) / n
    result["tick_ms_median"] = tick_times_ms[n // 2]
    result["tick_ms_p90"] = tick_times_ms[int(n * 0.9)]
    result["tick_ms_min"] = tick_times_ms[0]
    result["tick_ms_max"] = tick_times_ms[-1]
    print(f"[timing] mean={result['tick_ms_mean']:.2f}ms median={result['tick_ms_median']:.2f}ms "
          f"p90={result['tick_ms_p90']:.2f}ms min={result['tick_ms_min']:.2f}ms max={result['tick_ms_max']:.2f}ms")

    # cleanup actors (not the server process itself)
    for cam in cameras.values():
        cam.stop()
        cam.destroy()
    vehicle.destroy()

    settings.synchronous_mode = False
    world.apply_settings(settings)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
