# Part 1.3 Benchmark Analysis

## Setup

I ran the benchmark with `mpirun --oversubscribe -n 4 python part1/benchmark.py`, using 4 MPI ranks. The reported numbers are average milliseconds per forward pass.

| Workload | batch | input dim | hidden dim | topk | Simple MoE | TP MoE | EP MoE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small | 32 | 64 | 256 | 2 | 0.44 ms | 0.88 ms | 0.27 ms |
| Medium | 64 | 64 | 256 | 2 | 0.79 ms | 1.69 ms | 0.36 ms |
| Large | 32 | 128 | 512 | 2 | 1.40 ms | 1.48 ms | 0.57 ms |

## Discussion

Expert parallelism was fastest for all measured workloads. In EP, each rank owns one complete expert. Tokens are routed to the expert owner with all-to-all, processed locally, then sent back and combined. This makes EP communication-bound for small workloads because all-to-all latency is significant, but the computation is distributed across ranks, so it scales well as batch size or hidden dimension grows.

Tensor parallelism was slower for the small and medium workloads. In TP, every rank holds a shard of every expert, and each `ShardedLinear` needs an `Allreduce` to assemble the full output. Since the implementation routes one token at a time through each selected expert, TP performs many small collectives. For small matrix multiplications, the communication overhead dominates, so TP is communication-bound.

For the larger workload, TP became closer to the serial baseline: 1.48 ms vs. 1.40 ms. This happens because increasing the input and hidden dimensions increases the matrix multiplication work, so the cost of communication is better amortized. TP is still affected by collective overhead, but it shifts more toward compute-bound behavior as the workload grows.

Overall, EP is better for these benchmark settings because it pays a smaller routing cost and distributes full expert computation across ranks. TP would likely become more attractive only for much larger expert matrices or a more batched implementation that reduces the number of small collectives.
