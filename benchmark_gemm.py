import time
import torch
import tabulate
from triton.testing import do_bench
import torch.nn.functional as F

torch.manual_seed(0)
repeats = 100
warmup = 30
dtype = torch.bfloat16
device = 'cuda'
verbose = False

shapes = [
    (16384, 8192, 1280),
    (16384, 1024, 8192),
    (16384, 8192, 7168),
    (16384, 3584, 8192),
    (8192, 8192, 8192)
]

results = []

for (m, n, k) in shapes:
    # Matmul benchmark
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(n, k, device=device, dtype=dtype).transpose(-1, -2)
    nFLOPS = 2 * m * n * k
    ms = do_bench(lambda: torch.matmul(a, b), warmup=warmup, rep=repeats)
    tflops_matmul = nFLOPS / ms * 1e-9
    time.sleep(3) # reduce power throttling

    # Linear (without bias) benchmark using F.linear
    weight_no_bias = torch.randn(n, k, device=device, dtype=dtype)
    input_tensor = torch.randn(m, k, device=device, dtype=dtype)
    with torch.inference_mode():
        ms_linear_no_bias = do_bench(lambda: F.linear(input_tensor, weight_no_bias, bias=None), warmup=warmup, rep=repeats)
    tflops_linear_no_bias = nFLOPS / ms_linear_no_bias * 1e-9
    time.sleep(3) # reduce power throttling

    # addmm benchmark (a @ b + c)
    a = torch.randn(m, k, device=device, dtype=dtype)  # (m, k) shape
    b = torch.randn(n, k, device=device, dtype=dtype).t()  # (k, n) shape
    c = torch.randn(m, n, device=device, dtype=dtype)  # (m, n) shape
    nFLOPS_with_bias = 2 * m * n * k + m * n  # FLOPs for matmul and addition
    ms_addmm = do_bench(lambda: torch.addmm(c, a, b), warmup=warmup, rep=repeats)
    tflops_addmm = nFLOPS_with_bias / ms_addmm * 1e-9
    time.sleep(3) # reduce power throttling

    # Linear (with bias) benchmark using F.linear
    weight_with_bias = torch.randn(n, k, device=device, dtype=dtype)
    bias = torch.randn(n, device=device, dtype=dtype)
    input_tensor = torch.randn(m, k, device=device, dtype=dtype)
    with torch.inference_mode():
        ms_linear_with_bias = do_bench(lambda: F.linear(input_tensor, weight_with_bias, bias=bias), warmup=warmup, rep=repeats)
    tflops_linear_with_bias = nFLOPS_with_bias / ms_linear_with_bias * 1e-9
    time.sleep(3) # reduce power throttling
    
    # do F.linear with autocast bf16 with a b and c being fp32
    a = torch.randn(m, k, device=device, dtype=torch.float32)
    b = torch.randn(n, k, device=device, dtype=torch.float32).transpose(-1, -2)
    c = torch.randn(m, n, device=device, dtype=torch.float32)
    
    
    with torch.autocast(dtype=dtype):
        ms_autocast = do_bench(lambda: F.linear(a, b, bias=c), warmup=warmup, rep=repeats)
    tflops_autocast = nFLOPS_with_bias / ms_autocast * 1e-9

    # Append the results to the list
    results.append([f"({m}, {n}, {k})", f"{tflops_matmul:.1f} TFLOPS", f"{tflops_linear_no_bias:.1f} TFLOPS", 
                    f"{tflops_addmm:.1f} TFLOPS", f"{tflops_linear_with_bias:.1f} TFLOPS", f"{tflops_autocast:.1f} TFLOPS"])

# Print results using tabulate
headers = ["Shape (M, N, K)", "bf16 torch.matmul", "bf16 F.linear (without bias)", "bf16 torch.addmm", "bf16 F.linear (with bias)", "bf16 F.linear (with bias & amp)"]
print(f"Benchmark results for Realistic GEMM shapes with {warmup=} and {repeats=}")
print(tabulate.tabulate(results, headers=headers, tablefmt="grid"))