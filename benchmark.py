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


def _run(python_bin: str, backend: str, iters: int, warmup: int, device: str | None = None) -> dict:
    cmd = [python_bin, str(WORKER), "--backend", backend, "--iters", str(iters), "--warmup", str(warmup)]
    if device:
        cmd += ["--device", device]
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
    parser.add_argument("--real-torch-device", default="cpu", choices=["cpu", "mps"])
    args = parser.parse_args()

    print("Running torch-mlx backend...")
    mlx_result = _run(args.python, "torch-mlx", args.iters, args.warmup)

    print(f"Running real-torch backend (device={args.real_torch_device})...")
    torch_result = _run(args.python, "real-torch", args.iters, args.warmup, args.real_torch_device)

    _summarize("torch-mlx (this repo)", mlx_result)
    _summarize(f"real PyTorch ({args.real_torch_device})", torch_result)

    mlx_median = statistics.median(mlx_result["times_s"])
    torch_median = statistics.median(torch_result["times_s"])
    ratio = torch_median / mlx_median
    faster = "torch-mlx" if ratio > 1 else f"real PyTorch ({args.real_torch_device})"
    print(f"\n{faster} is {max(ratio, 1 / ratio):.2f}x faster (median forward-pass time)")


if __name__ == "__main__":
    main()
