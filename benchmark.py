"""Speed comparison: this repo's DepthAnythingMLX (torch-mlx) vs real,
unmodified PyTorch + transformers ("the standard repo") -- same model
(depth-anything/Depth-Anything-V2-Large-hf), same real weights, same
input, same number of forward passes. Runs each backend in its own
subprocess (real torch and torch-mlx can't share a process).

usage: python3 benchmark.py [--iters 10] [--warmup 2] [--real-torch-device cpu|mps]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = HERE / "_bench_worker.py"


def _run(python_bin: str, backend: str, iters: int, warmup: int, device: str | None = None, compiled: bool = False, image_size: tuple[int, int] = (480, 640)) -> dict:
    cmd = [python_bin, str(WORKER), "--backend", backend, "--iters", str(iters), "--warmup", str(warmup),
           "--height", str(image_size[0]), "--width", str(image_size[1])]
    if device:
        cmd += ["--device", device]
    if compiled:
        cmd += ["--compiled"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{backend} benchmark failed:\n{result.stderr[-3000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _summarize(label: str, data: dict) -> None:
    times = data["times_s"]
    print(f"\n{label} (device={data['device']})")
    print(f"  model load: {data['load_s']:.1f}s")
    print(f"  forward pass over {len(times)} iters:")
    print(f"    mean:   {statistics.mean(times) * 1000:.1f} ms")
    print(f"    median: {statistics.median(times) * 1000:.1f} ms")
    print(f"    min:    {min(times) * 1000:.1f} ms")
    print(f"    max:    {max(times) * 1000:.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable, help="interpreter with real torch + transformers + mlx installed")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--all", action="store_true",
        help="run all three: torch-mlx, real PyTorch CPU, real PyTorch MPS -- instead of --real-torch-device's single choice",
    )
    parser.add_argument("--real-torch-device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--height", type=int, default=480, help="test image height (note: DPTImageProcessor resizes to ~518px regardless, so this mainly stresses preprocessing/resize cost, not model compute)")
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    size = (args.height, args.width)

    runs = []
    print(f"Running torch-mlx backend (image {size[1]}x{size[0]})...")
    runs.append(("torch-mlx", _run(args.python, "torch-mlx", args.iters, args.warmup, image_size=size)))

    print("Running torch-mlx backend (mx.compile)...")
    runs.append(("torch-mlx (compiled)", _run(args.python, "torch-mlx", args.iters, args.warmup, compiled=True, image_size=size)))

    if args.all:
        for device in ("cpu", "mps"):
            print(f"Running real-torch backend (device={device})...")
            runs.append((f"real PyTorch ({device})", _run(args.python, "real-torch", args.iters, args.warmup, device, image_size=size)))
    else:
        print(f"Running real-torch backend (device={args.real_torch_device})...")
        runs.append((
            f"real PyTorch ({args.real_torch_device})",
            _run(args.python, "real-torch", args.iters, args.warmup, args.real_torch_device, image_size=size),
        ))

    for label, result in runs:
        _summarize(label, result)

    medians = {label: statistics.median(result["times_s"]) for label, result in runs}
    fastest_label = min(medians, key=medians.get)
    print(f"\n{'label':<22}{'median ms':>12}{'vs fastest':>14}")
    for label, median in sorted(medians.items(), key=lambda kv: kv[1]):
        ratio = median / medians[fastest_label]
        print(f"{label:<22}{median * 1000:>12.1f}{ratio:>13.2f}x")
    print(f"\nFastest: {fastest_label}")


if __name__ == "__main__":
    main()
