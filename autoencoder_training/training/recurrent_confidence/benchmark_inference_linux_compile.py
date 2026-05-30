from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import json
import platform
import torch
import numpy as np
from PIL import Image
import torch._inductor.config as inductor_config


from RecurrentDenoisingAutoencoder import RecurrentDenoisingAutoencoder, _make_channels


CHECKPOINT_PATH = Path("../../final_models/recurrent_confidence_closed_rooms_C24/autoencoder_best.pt")
BENCHMARK_SEQUENCE_PATH = Path("../../benchmark_sequence")

IN_CHANNELS  = 5
OUT_CHANNELS = 3
BASE         = 24

WARMUP_RUNS = 5

SAVE_DEBUG_IMAGES = False
DEBUG_IMAGE_DIR = Path("./debug_compile_outputs")

OUTPUT_JSON_PATH = Path("./benchmark_results.json")


def zero_hidden(batch_size, base, h, w, device):
    C = _make_channels(base, 5)
    return (
        torch.zeros(batch_size, C[0], h,      w,      device=device),
        torch.zeros(batch_size, C[1], h >> 1, w >> 1, device=device),
        torch.zeros(batch_size, C[2], h >> 2, w >> 2, device=device),
        torch.zeros(batch_size, C[3], h >> 3, w >> 3, device=device),
        torch.zeros(batch_size, C[4], h >> 4, w >> 4, device=device),
    )


def load_frame(seq_dir, frame_idx):
    frame = f"{frame_idx:04d}"

    color = np.asarray(
        Image.open(seq_dir / "input" / f"{frame}.png").convert("RGB"),
        dtype=np.float32,
    ) / 255.0

    depth = np.load(seq_dir / "depth" / f"{frame}.npy")

    conf_path = seq_dir / "confidence" / f"{frame}.npy"
    conf = np.load(conf_path) if conf_path.exists() else np.zeros(depth.shape, dtype=np.float32)

    color_t = torch.from_numpy(color).permute(2, 0, 1)
    depth_t = torch.from_numpy(depth).unsqueeze(0)
    conf_t  = torch.from_numpy(conf).unsqueeze(0)

    return torch.cat([color_t, depth_t, conf_t], dim=0)


def tensor_to_image_uint8(y):
    """
    Converts model output [1, 3, H, W] or [3, H, W] to uint8 RGB image.
    Assumes output is roughly in [0, 1].
    """
    if y.ndim == 4:
        y = y[0]

    y = y.detach().float().cpu()
    y = torch.clamp(y, 0.0, 1.0)
    y = y.permute(1, 2, 0).numpy()
    y = (y * 255.0).round().astype(np.uint8)

    return y


def summarize_times(frame_times):
    times = np.array(frame_times, dtype=np.float64)
    total = float(times.sum())

    return {
        "frames": int(len(times)),
        "total_ms": float(total * 1000.0),
        "mean_ms_per_frame": float(times.mean() * 1000.0),
        "median_ms_per_frame": float(np.median(times) * 1000.0),
        "std_ms": float(times.std() * 1000.0),
        "min_ms": float(times.min() * 1000.0),
        "max_ms": float(times.max() * 1000.0),
        "throughput_fps": float(len(times) / total) if total > 0 else 0.0,
        "frame_times_ms": [float(t * 1000.0) for t in times],
    }


def print_stats(name, stats):
    print(f"\n{'=' * 40}")
    print(f"  {name}")
    print(f"{'=' * 40}")
    print(f"  Frames     : {stats['frames']}")
    print(f"  Total time : {stats['total_ms']:.1f} ms")
    print(f"  Mean       : {stats['mean_ms_per_frame']:.2f} ms/frame")
    print(f"  Median     : {stats['median_ms_per_frame']:.2f} ms/frame")
    print(f"  Std        : {stats['std_ms']:.2f} ms")
    print(f"  Min        : {stats['min_ms']:.2f} ms")
    print(f"  Max        : {stats['max_ms']:.2f} ms")
    print(f"  Throughput : {stats['throughput_fps']:.1f} FPS")
    print(f"{'=' * 40}")


@torch.inference_mode()
def benchmark_model(model, model_name, preloaded, base, pH, pW, device):
    print(f"\nWarmup: {model_name} ({WARMUP_RUNS} runs)...")

    for _ in range(WARMUP_RUNS):
        h1, h2, h3, h4, h5 = zero_hidden(1, base, pH, pW, device)

        for x in preloaded:
            _, h1, h2, h3, h4, h5 = model(x, h1, h2, h3, h4, h5)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"Benchmarking: {model_name}...")

    h1, h2, h3, h4, h5 = zero_hidden(1, base, pH, pW, device)
    frame_times = []

    for t, x in enumerate(preloaded):
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        _, h1, h2, h3, h4, h5 = model(x, h1, h2, h3, h4, h5)

        if device.type == "cuda":
            torch.cuda.synchronize()

        frame_times.append(time.perf_counter() - t0)

    stats = summarize_times(frame_times)
    print_stats(model_name, stats)

    return stats


@torch.inference_mode()
def compare_eager_vs_compiled(eager_model, compiled_model, preloaded, base, pH, pW, device):
    print("\nChecking eager vs compiled output correctness...")

    if SAVE_DEBUG_IMAGES:
        DEBUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    h1_e, h2_e, h3_e, h4_e, h5_e = zero_hidden(1, base, pH, pW, device)
    h1_c, h2_c, h3_c, h4_c, h5_c = zero_hidden(1, base, pH, pW, device)

    max_abs_errors = []
    mean_abs_errors = []
    rmses = []

    frames_to_save = {
        0,
        len(preloaded) // 2,
        len(preloaded) - 1,
    }

    for t, x in enumerate(preloaded):
        y_e, h1_e, h2_e, h3_e, h4_e, h5_e = eager_model(
            x, h1_e, h2_e, h3_e, h4_e, h5_e
        )

        y_c, h1_c, h2_c, h3_c, h4_c, h5_c = compiled_model(
            x, h1_c, h2_c, h3_c, h4_c, h5_c
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        diff = (y_e - y_c).detach().float()

        max_abs = diff.abs().max().item()
        mean_abs = diff.abs().mean().item()
        rmse = torch.sqrt(torch.mean(diff * diff)).item()

        max_abs_errors.append(max_abs)
        mean_abs_errors.append(mean_abs)
        rmses.append(rmse)

        print(
            f"  Frame {t:04d}: "
            f"max_abs={max_abs:.8f}, "
            f"mean_abs={mean_abs:.8f}, "
            f"rmse={rmse:.8f}"
        )

        if SAVE_DEBUG_IMAGES and t in frames_to_save:
            eager_img = tensor_to_image_uint8(y_e)
            compiled_img = tensor_to_image_uint8(y_c)

            abs_diff = (y_e - y_c).abs()
            abs_diff = abs_diff / (abs_diff.max() + 1e-8)
            diff_img = tensor_to_image_uint8(abs_diff)

            Image.fromarray(eager_img).save(DEBUG_IMAGE_DIR / f"frame_{t:04d}_eager.png")
            Image.fromarray(compiled_img).save(DEBUG_IMAGE_DIR / f"frame_{t:04d}_compiled.png")
            Image.fromarray(diff_img).save(DEBUG_IMAGE_DIR / f"frame_{t:04d}_diff_amplified.png")

    max_abs_errors = np.array(max_abs_errors, dtype=np.float64)
    mean_abs_errors = np.array(mean_abs_errors, dtype=np.float64)
    rmses = np.array(rmses, dtype=np.float64)

    correctness = {
        "frames": int(len(preloaded)),
        "max_absolute_error": float(max_abs_errors.max()),
        "mean_absolute_error": float(mean_abs_errors.mean()),
        "mean_rmse": float(rmses.mean()),
        "per_frame": [
            {
                "frame": int(i),
                "max_absolute_error": float(max_abs_errors[i]),
                "mean_absolute_error": float(mean_abs_errors[i]),
                "rmse": float(rmses[i]),
            }
            for i in range(len(preloaded))
        ],
    }

    print("\nEager vs compiled summary:")
    print(f"  Max absolute error : {correctness['max_absolute_error']:.10f}")
    print(f"  Mean absolute error: {correctness['mean_absolute_error']:.10f}")
    print(f"  Mean RMSE          : {correctness['mean_rmse']:.10f}")

    if SAVE_DEBUG_IMAGES:
        print(f"  Debug images saved to: {DEBUG_IMAGE_DIR.resolve()}")

    print("Correctness check complete.\n")

    return correctness


def build_model(in_channels, out_channels, base, ckpt, device):
    model = RecurrentDenoisingAutoencoder(
        in_channels=in_channels,
        out_channels=out_channels,
        base=base,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model


@torch.inference_mode()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    in_channels  = ckpt.get("in_channels",   IN_CHANNELS)
    out_channels = ckpt.get("out_channels",  OUT_CHANNELS)
    base         = ckpt.get("base_channels", BASE)

    print(f"Loaded checkpoint (epoch {ckpt['epoch']})")

    eager_model = build_model(
        in_channels=in_channels,
        out_channels=out_channels,
        base=base,
        ckpt=ckpt,
        device=device,
    )

    compiled_model = None
    compile_success = False
    compile_error = None

    if device.type == "cuda":
        print("Compiling model without CUDA Graphs...")

        try:
            compiled_base_model = build_model(
                in_channels=in_channels,
                out_channels=out_channels,
                base=base,
                ckpt=ckpt,
                device=device,
            )

            compiled_model = torch.compile(
                compiled_base_model,
                fullgraph=False,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )

            compile_success = True
            print("Model compiled.")

        except Exception as e:
            compile_error = str(e)
            compiled_model = None
            compile_success = False
            print(f"torch.compile failed: {e}")

    else:
        compile_error = "CUDA is not available; compiled benchmark skipped."

    seq_dir = BENCHMARK_SEQUENCE_PATH
    frames = sorted((seq_dir / "input").glob("*.png"))
    T = len(frames)

    print(f"Sequence: {seq_dir.name} ({T} frames)")

    print("Preloading frames to GPU...")
    t0 = time.perf_counter()

    preloaded = [
        load_frame(seq_dir, t).unsqueeze(0).to(device)
        for t in range(T)
    ]

    load_time = time.perf_counter() - t0
    print(f"  Preloaded {T} frames in {load_time:.3f}s")

    _, _, H, W = preloaded[0].shape
    pH = (H + 31) // 32 * 32
    pW = (W + 31) // 32 * 32

    print(f"  Resolution: {H}x{W} (padded to {pH}x{pW})")

    metadata = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_hip_version": getattr(torch.version, "hip", None),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "sequence_path": str(BENCHMARK_SEQUENCE_PATH),
        "sequence_name": seq_dir.name,
        "frames": int(T),
        "input_resolution": {
            "height": int(H),
            "width": int(W),
        },
        "padded_resolution": {
            "height": int(pH),
            "width": int(pW),
        },
        "warmup_runs": int(WARMUP_RUNS),
        "preload_time_seconds": float(load_time),
        "in_channels": int(in_channels),
        "out_channels": int(out_channels),
        "base_channels": int(base),
        "compile": {
            "attempted": bool(device.type == "cuda"),
            "success": bool(compile_success),
            "error": compile_error,
            "fullgraph": False,
            "dynamic": False,
            "cudagraphs_disabled": True,
            "options": {
                "triton.cudagraphs": False,
            },
        },
    }

    results = {
        "metadata": metadata,
        "benchmarks": {},
        "correctness": None,
        "speedup": None,
    }

    if device.type == "cuda":
        torch.cuda.synchronize()

    eager_stats = benchmark_model(
        model=eager_model,
        model_name="eager",
        preloaded=preloaded,
        base=base,
        pH=pH,
        pW=pW,
        device=device,
    )

    results["benchmarks"]["eager"] = eager_stats

    if compiled_model is not None:
        if device.type == "cuda":
            torch.cuda.synchronize()

        compiled_stats = benchmark_model(
            model=compiled_model,
            model_name="compiled",
            preloaded=preloaded,
            base=base,
            pH=pH,
            pW=pW,
            device=device,
        )

        results["benchmarks"]["compiled"] = compiled_stats

        correctness = compare_eager_vs_compiled(
            eager_model=eager_model,
            compiled_model=compiled_model,
            preloaded=preloaded,
            base=base,
            pH=pH,
            pW=pW,
            device=device,
        )

        results["correctness"] = correctness

        eager_mean = eager_stats["mean_ms_per_frame"]
        compiled_mean = compiled_stats["mean_ms_per_frame"]

        results["speedup"] = {
            "mean_ms_per_frame_ratio_eager_over_compiled": (
                float(eager_mean / compiled_mean) if compiled_mean > 0 else None
            ),
            "fps_ratio_compiled_over_eager": (
                float(compiled_stats["throughput_fps"] / eager_stats["throughput_fps"])
                if eager_stats["throughput_fps"] > 0
                else None
            ),
            "mean_ms_per_frame_delta": float(eager_mean - compiled_mean),
            "throughput_fps_delta": float(
                compiled_stats["throughput_fps"] - eager_stats["throughput_fps"]
            ),
        }

        print("\nSpeedup summary:")
        print(
            "  Mean frame time speedup: "
            f"{results['speedup']['mean_ms_per_frame_ratio_eager_over_compiled']:.3f}x"
        )
        print(
            "  FPS speedup            : "
            f"{results['speedup']['fps_ratio_compiled_over_eager']:.3f}x"
        )
        print(
            "  Mean frame time delta  : "
            f"{results['speedup']['mean_ms_per_frame_delta']:.3f} ms/frame"
        )
        print(
            "  FPS delta              : "
            f"{results['speedup']['throughput_fps_delta']:.3f} FPS"
        )

    else:
        print("\nCompiled benchmark skipped because compilation failed.")

    OUTPUT_JSON_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved benchmark JSON to: {OUTPUT_JSON_PATH.resolve()}")


if __name__ == "__main__":
    main()