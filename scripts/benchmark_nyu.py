"""Standalone random-weight NYU benchmark; no Apex, downloads or sibling imports."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import tarfile
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms as T

from scripts.benchmark_adapter import (
    MODEL_NAME, DCN_BACKEND, model_config, build_model, DepthOnly, reference_prediction,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output(command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return (result.stdout + result.stderr).strip()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


class NYUInputs:
    """Match the previous SparseDC evaluation transform and per-index sampling."""

    def __init__(self, root, seed=2023):
        self.root, self.seed = Path(root), seed
        self.samples = json.loads((self.root / "nyu_h5_pairs.json").read_text())["test"]
        self.rgb_transform = T.Compose([
            T.Resize(240), T.CenterCrop((228, 304)), T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.depth_transform = T.Compose([T.Resize(240), T.CenterCrop((228, 304)), T.ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        with h5py.File(self.root / self.samples[index]["filename"], "r") as sample:
            rgb = sample["rgb"][:].transpose(1, 2, 0).astype("uint8")
            depth = sample["depth"][:].astype("float32")
        rgb = self.rgb_transform(Image.fromarray(rgb, mode="RGB"))
        depth = self.depth_transform(Image.fromarray(depth, mode="F")).clamp(0.0, 10.0)
        valid = torch.nonzero(depth.reshape(-1) > 1e-4, as_tuple=False).flatten()
        generator = torch.Generator().manual_seed(self.seed + index)
        picked = valid[torch.randperm(valid.numel(), generator=generator)[:500]]
        mask = torch.zeros(depth.numel(), dtype=depth.dtype)
        mask[picked] = 1.0
        dep = depth * mask.reshape_as(depth)
        return {"rgb": rgb.unsqueeze(0), "dep": dep.unsqueeze(0)}


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean_ms": float(values.mean()),
            "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "std_ms": float(values.std()), "min_ms": float(values.min()),
            "max_ms": float(values.max()), "fps": float(1000 / values.mean())}


def macs(model, rgb, dep):
    from fvcore.nn.jit_analysis import JitModelAnalysis
    from fvcore.nn.jit_handles import (
        bmm_flop_jit, conv_flop_jit, einsum_flop_jit, get_shape, linear_flop_jit, matmul_flop_jit,
    )
    calls = []

    def dcn(inputs, outputs):
        weight, result = get_shape(inputs[1]), get_shape(outputs[0])
        count = math.prod(result) * math.prod(weight[1:])
        calls.append({"weight_shape": weight, "output_shape": result, "macs": count})
        return Counter({"deform_conv_accumulation": count})

    analysis = JitModelAnalysis(model, (rgb, dep)).set_op_handle(**{
        "aten::_convolution": conv_flop_jit, "aten::linear": linear_flop_jit,
        "aten::matmul": matmul_flop_jit, "aten::mm": matmul_flop_jit,
        "aten::bmm": bmm_flop_jit, "aten::einsum": einsum_flop_jit,
        "torchvision::deform_conv2d": dcn,
    }).unsupported_ops_warnings(False).uncalled_modules_warnings(False)
    count = analysis.total()
    return {"macs": int(count), "macs_g": count / 1e9,
            "by_operator": dict(analysis.by_operator()),
            "excluded_operator_calls": dict(analysis.unsupported_ops()),
            "uncalled_modules": sorted(analysis.uncalled_modules()), "dcn_calls": calls,
            "definition": "conv (including transpose), linear, matmul and nominal DCN accumulation; excludes interpolation/modulation, bias, normalization, pooling and elementwise ops"}


class GpuMonitor:
    def __init__(self, path):
        self.path, self.stop = path, threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        fields = "timestamp,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem"
        with self.path.open("w") as stream:
            stream.write(fields + "\n")
            while not self.stop.is_set():
                stream.write(output(["nvidia-smi", "--query-gpu=" + fields,
                                     "--format=csv,noheader,nounits"]) + "\n")
                stream.flush()
                self.stop.wait(1.0)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def benchmark(model, dataset, args, directory):
    rows, groups = [], []
    with GpuMonitor(directory / "gpu_during_speed.csv"), torch.inference_mode():
        for group in range(args.groups):
            index = round(group * (len(dataset) - 1) / max(args.groups - 1, 1))
            sample = dataset[index]
            hashes = {key: hashlib.sha256(value.numpy().tobytes()).hexdigest()
                      for key, value in sample.items()}
            rgb, dep = (sample[key].cuda() for key in ("rgb", "dep"))
            del sample
            for _ in range(args.warmup):
                model(rgb, dep)
            torch.cuda.synchronize()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            current = []
            for iteration in range(args.iterations):
                wall_start = time.perf_counter()
                start.record()
                pred = model(rgb, dep)
                end.record()
                end.synchronize()
                wall_ms = (time.perf_counter() - wall_start) * 1000
                row = [group + 1, index, iteration + 1, start.elapsed_time(end), wall_ms]
                del pred
                rows.append(row)
                current.append(row)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            pred = model(rgb, dep)
            torch.cuda.synchronize()
            memory = {"baseline_allocated_mib": baseline / 2**20,
                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                      "incremental_peak_mib": (torch.cuda.max_memory_allocated() - baseline) / 2**20,
                      "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
            if not torch.isfinite(pred).all() or pred.shape != dep.shape:
                raise ValueError("Nonfinite or incorrectly shaped depth prediction")
            del pred, rgb, dep
            measured = summary([row[3] for row in current])
            groups.append({"group": group + 1, "sample_index": index, "sample": dataset.samples[index],
                           "input_sha256": hashes, "cuda_events": measured,
                           "wall": summary([row[4] for row in current]), "memory": memory})
            print(f"Group {group + 1}: {measured['mean_ms']:.3f} ms, P95 {measured['p95_ms']:.3f} ms", flush=True)
    with (directory / "latency.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "sample_index", "iteration", "cuda_event_ms", "synchronized_wall_ms"])
        writer.writerows(rows)
    return {"groups": groups, "cuda_events": summary([row[3] for row in rows]),
            "wall": summary([row[4] for row in rows]),
            "peak_allocated_mib": max(group["memory"]["peak_allocated_mib"] for group in groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/nyudepthv2_h5")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--groups", type=int, default=3)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.groups) <= 0:
        parser.error("warmup, iterations and groups must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    args.output.mkdir(parents=True, exist_ok=False)
    directory = args.output
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    config = model_config(args.seed, args.data_dir)
    net = build_model(config)
    params = sum(value.numel() for value in net.parameters())
    frozen = sum(value.numel() for value in net.parameters() if not value.requires_grad)
    weights = directory / f"{MODEL_NAME.lower()}_nyu_random_fp32_state_dict.pt"
    torch.save(net.state_dict(), weights)
    state_bytes = sum(value.numel() * value.element_size() for value in net.state_dict().values())
    net = net.float().cuda().eval()
    model = DepthOnly(net).float().cuda().eval()
    dataset = NYUInputs(args.data_dir, args.seed)
    sources = [path for path in ROOT.rglob('*.py')
               if not {'outputs', '.git', '__pycache__'}.intersection(path.relative_to(ROOT).parts)]
    write_json(directory / "source_sha256.json", {str(path.relative_to(ROOT)): sha256(path) for path in sources})
    with tarfile.open(directory / 'source_snapshot.tar.gz', 'w:gz') as archive:
        for path in sources + [ROOT / 'environment.yaml', ROOT / 'BENCHMARK.md', ROOT / '.gitignore']:
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(ROOT)))
    (directory / "git_diff.patch").write_text(output(["git", "diff", "--ignore-space-at-eol", "HEAD"]))
    (directory / "pip_freeze.txt").write_text(output([sys.executable, "-m", "pip", "freeze"]))
    (directory / "conda_explicit.txt").write_text(output(["conda", "list", "--prefix", sys.prefix, "--explicit"]))
    write_json(directory / "model_config.json", vars(config))
    packages = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "pillow", "h5py", "fvcore")}
    record = {
        "status": "in_progress", "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"name": MODEL_NAME, "weights": "random initialization; no checkpoint/pretrained weights", "seed": args.seed,
                  "prop_time": config.prop_time, "config": vars(config)},
        "commit": output(["git", "rev-parse", "HEAD"]), "git_status": output(["git", "status", "--short"]),
        "command": sys.argv,
        "data": {"path": str(args.data_dir.resolve()), "split_sha256": sha256(args.data_dir / "nyu_h5_pairs.json"),
                 "test_samples": len(dataset), "input_hw": [228, 304], "num_sample": 500,
                 "seed": args.seed, "sampling": "deterministic_per_index (seed + index)"},
        "protocol": {"batch_size": 1, "dtype": "FP32", "tf32_matmul": False, "tf32_cudnn": False,
                     "autocast": False, "cudnn_benchmark": False, "cudnn_deterministic": False,
                     "dcn_backend": DCN_BACKEND, "inference_mode": True, "compile": False, "cuda_graphs": False,
                     "warmup_per_group": args.warmup, "iterations_per_group": args.iterations, "groups": args.groups,
                     "torch_cpu_threads": torch.get_num_threads(),
                     "timer": "per-frame CUDA events, synchronize each end event; also synchronized host wall time",
                     "excluded": ["data loading", "H2D transfers", "GT", "losses", "metrics", "logging", "visualization"]},
        "environment": {"python": sys.version, "python_executable": sys.executable, "conda_prefix": sys.prefix,
                        "platform": platform.platform(), "packages": packages, "cuda_runtime": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
                        "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
                        "nvidia_smi_before": output(["nvidia-smi"])},
        "size": {"parameters": params, "trainable_parameters": params - frozen, "frozen_parameters": frozen,
                 "parameter_fp32_mib": params * 4 / 2**20, "state_tensor_bytes": state_bytes,
                 "state_tensor_mib": state_bytes / 2**20, "state_dict_file_mib": weights.stat().st_size / 2**20,
                 "state_dict_file_sha256": sha256(weights)}, "accuracy": None,
    }
    equivalence = []
    with torch.inference_mode():
        for index in (0, len(dataset) // 2, len(dataset) - 1):
            sample = {key: value.cuda() for key, value in dataset[index].items()}
            reference = reference_prediction(model, sample['rgb'], sample['dep'])
            actual = model(sample["rgb"], sample["dep"])
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
            assert torch.isfinite(actual).all() and actual.shape == sample["dep"].shape
            equivalence.append({"index": index, "max_abs_error": float((reference - actual).abs().max()),
                                "bitwise_equal": torch.equal(reference, actual),
                                "sparse_points": int((sample["dep"] > 0).sum())})
            del reference, actual, sample
        record["equivalence"] = equivalence
        sample = {key: value.cuda() for key, value in dataset[0].items()}
        try:
            record["complexity"] = macs(model, sample["rgb"], sample["dep"])
        except Exception as error:
            record["complexity"] = {'macs': None, 'macs_g': None, 'error': repr(error)}
        del sample
    write_json(directory / "results.json", record)
    print("Random initialization, finite outputs and three-sample equivalence verified", flush=True)
    torch.cuda.empty_cache()
    record["speed"] = benchmark(model, dataset, args, directory)
    record["status"] = "complete"
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["environment"]["nvidia_smi_after"] = output(["nvidia-smi"])
    write_json(directory / "results.json", record)
    print(json.dumps({"model": MODEL_NAME, "size": record["size"], "macs_g": record["complexity"]["macs_g"],
                      "speed": record["speed"]["cuda_events"], "peak_allocated_mib": record["speed"]["peak_allocated_mib"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
