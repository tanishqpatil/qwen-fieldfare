import os
import sys
import time
import subprocess
import resource
import json

def get_swap_usage():
    try:
        out = subprocess.check_output(["sysctl", "vm.swapusage"]).decode("utf-8")
        # vm.swapusage: total = 2048.00M  used = 767.56M  free = 1280.44M  (encrypted)
        parts = out.split()
        used = parts[6] # '767.56M'
        return used
    except Exception as e:
        return str(e)

def get_rss_mb():
    # maxrss on macOS is in bytes
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / (1024 * 1024)

print("=== FLINT ALL-RESIDENT MLX BENCHMARK ===")
print("Python:", sys.executable)
import mlx.core as mx
import mlx_vlm
from mlx_vlm import load, generate

print("MLX version:", mx.__version__)
print("mlx_vlm version:", getattr(mlx_vlm, "__version__", "unknown"))
print("Initial Swap:", get_swap_usage())
print("Initial RSS: {:.2f} MB".format(get_rss_mb()))

model_path = "/Users/tanishq/models/flint-mixed-3-4bit"
print("Model path:", model_path)

# --- SCENARIO 1: MODEL LOAD ---
t0 = time.perf_counter()
print("\n--- Loading Model (Cold) ---")
mx.reset_peak_memory()
model, processor = load(model_path)
mx.eval(model.parameters())
load_time = time.perf_counter() - t0
peak_load_mem_gb = mx.get_peak_memory() / (1024**3)
active_load_mem_gb = mx.get_active_memory() / (1024**3)
rss_load_gb = get_rss_mb() / 1024

print(f"Model Load Time: {load_time:.2f} s")
print(f"MLX Active Memory: {active_load_mem_gb:.2f} GB")
print(f"MLX Peak Memory: {peak_load_mem_gb:.2f} GB")
print(f"Process Max RSS: {rss_load_gb:.2f} GB")
print("Swap after load:", get_swap_usage())

# --- SCENARIO 2: SHORT PROMPT (3 RUNS) ---
short_prompt = "Write a Swift function that implements a thread-safe bounded LRU cache."
print(f"\n--- Scenario 2: Short Prompt ({len(short_prompt)} chars) ---")

short_results = []
for run_idx in range(1, 4):
    print(f"\n--- Short Prompt Run {run_idx} ---")
    mx.reset_peak_memory()
    t_start = time.perf_counter()
    
    # Deterministic generation: temp=0.0
    res = generate(
        model,
        processor,
        prompt=short_prompt,
        max_tokens=64,
        temperature=0.0,
        verbose=False
    )
    total_time = time.perf_counter() - t_start
    
    peak_mem_gb = mx.get_peak_memory() / (1024**3)
    active_mem_gb = mx.get_active_memory() / (1024**3)
    rss_gb = get_rss_mb() / 1024
    
    prompt_tokens = getattr(res, "prompt_tokens", len(processor.encode(short_prompt)))
    gen_tokens = getattr(res, "generation_tokens", len(processor.encode(res.text)) if hasattr(res, "text") else 0)
    prompt_tps = getattr(res, "prompt_tps", 0.0)
    gen_tps = getattr(res, "generation_tps", 0.0)
    finish_reason = getattr(res, "finish_reason", "unknown")
    
    run_data = {
        "run": run_idx,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "finish_reason": finish_reason,
        "total_time_s": total_time,
        "prompt_tps": prompt_tps,
        "generation_tps": gen_tps,
        "mlx_active_gb": active_mem_gb,
        "mlx_peak_gb": peak_mem_gb,
        "rss_gb": rss_gb,
        "swap": get_swap_usage(),
        "text_preview": res.text[:120].replace("\n", " ") if hasattr(res, "text") else ""
    }
    short_results.append(run_data)
    print(f"  Prompt tokens: {prompt_tokens}, Gen tokens: {gen_tokens}")
    print(f"  Total time: {total_time:.3f} s")
    print(f"  Prompt throughput: {prompt_tps:.2f} tok/s")
    print(f"  Generation throughput: {gen_tps:.2f} tok/s")
    print(f"  Peak memory: {peak_mem_gb:.2f} GB, RSS: {rss_gb:.2f} GB")
    print(f"  Text: {run_data['text_preview']}")

# --- SCENARIO 3: LONG PROMPT (~8K TOKENS) ---
print("\n--- Scenario 3: Long Prompt (~8K Tokens) ---")

# Construct ~8K prompt from realistic code
base_code = """
// Swift implementation of an expert cache and matrix multiplication kernel
public struct MatrixMultiplier {
    public let rows: Int
    public let cols: Int
    public let data: [Float]
    public func multiply(by other: MatrixMultiplier) -> MatrixMultiplier {
        precondition(cols == other.rows, "Dimension mismatch")
        var result = [Float](repeating: 0, count: rows * other.cols)
        for i in 0..<rows {
            for k in 0..<cols {
                let a = data[i * cols + k]
                for j in 0..<other.cols {
                    result[i * other.cols + j] += a * other.data[k * other.cols + j]
                }
            }
        }
        return MatrixMultiplier(rows: rows, cols: other.cols, data: result)
    }
}
"""
# Tokenize base snippet to calibrate repeats
sample_tokens = processor.encode(base_code)
repeats = max(1, 8000 // max(1, len(sample_tokens)))
long_prompt = "Analyze the following Swift code for performance and provide recommendations:\n" + (base_code * repeats)
encoded_long = processor.encode(long_prompt)
print(f"Targeting ~8k tokens. Actual encoded prompt token count: {len(encoded_long)}")

long_results = []
for run_idx in range(1, 4):
    print(f"\n--- Long Prompt Run {run_idx} ---")
    mx.reset_peak_memory()
    t_start = time.perf_counter()
    
    res = generate(
        model,
        processor,
        prompt=long_prompt,
        max_tokens=32,
        temperature=0.0,
        verbose=False
    )
    total_time = time.perf_counter() - t_start
    
    peak_mem_gb = mx.get_peak_memory() / (1024**3)
    active_mem_gb = mx.get_active_memory() / (1024**3)
    rss_gb = get_rss_mb() / 1024
    
    prompt_tokens = getattr(res, "prompt_tokens", len(encoded_long))
    gen_tokens = getattr(res, "generation_tokens", len(processor.encode(res.text)) if hasattr(res, "text") else 0)
    prompt_tps = getattr(res, "prompt_tps", 0.0)
    gen_tps = getattr(res, "generation_tps", 0.0)
    finish_reason = getattr(res, "finish_reason", "unknown")
    
    run_data = {
        "run": run_idx,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "finish_reason": finish_reason,
        "total_time_s": total_time,
        "prompt_tps": prompt_tps,
        "generation_tps": gen_tps,
        "mlx_active_gb": active_mem_gb,
        "mlx_peak_gb": peak_mem_gb,
        "rss_gb": rss_gb,
        "swap": get_swap_usage(),
        "text_preview": res.text[:120].replace("\n", " ") if hasattr(res, "text") else ""
    }
    long_results.append(run_data)
    print(f"  Prompt tokens: {prompt_tokens}, Gen tokens: {gen_tokens}")
    print(f"  Total time: {total_time:.3f} s")
    print(f"  Prompt throughput: {prompt_tps:.2f} tok/s")
    print(f"  Generation throughput: {gen_tps:.2f} tok/s")
    print(f"  Peak memory: {peak_mem_gb:.2f} GB, RSS: {rss_gb:.2f} GB")
    print(f"  Swap: {get_swap_usage()}")

# Save complete results
output_summary = {
    "model_path": model_path,
    "mlx_version": mx.__version__,
    "mlx_vlm_version": getattr(mlx_vlm, "__version__", "unknown"),
    "model_load_time_s": load_time,
    "load_active_gb": active_load_mem_gb,
    "load_peak_gb": peak_load_mem_gb,
    "load_rss_gb": rss_load_gb,
    "short_prompt_runs": short_results,
    "long_prompt_runs": long_results
}

default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlx_flint_benchmark_results.json")
out_file = os.environ.get("FLINT_BENCHMARK_OUTPUT", sys.argv[1] if len(sys.argv) > 1 else default_out)
with open(out_file, "w") as f:
    json.dump(output_summary, f, indent=2)

print(f"\nBenchmark completed successfully! Results written to {out_file}")
