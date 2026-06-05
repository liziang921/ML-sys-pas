# Part 2.3 — Why MoE?

DeepSeek-V3 illustrates the main advantage of MoE: it can store far more
knowledge in total parameters than it activates for any one token. The model
has roughly 671B total parameters, but only a small subset of routed experts
plus the shared expert are active per token, giving an activated footprint on
the order of tens of billions of parameters rather than hundreds of billions.
Compared with a same-budget dense model, this can reduce per-token training
and inference FLOPs while still giving the model access to a much larger pool
of specialized expert weights.

The concrete advantage is inference and training efficiency at moderate load.
A dense model must run every MLP parameter for every token, while an MoE layer
runs only `topk` routed experts and the shared expert. This lets the model
increase total capacity without increasing activated compute proportionally,
which is why MoE models can be attractive when the workload has enough
parallelism to keep experts busy but not so much pressure that routing becomes
the bottleneck.

The concrete disadvantage is communication and load imbalance. Expert
parallelism requires all-to-all token exchange so tokens reach the ranks that
own their selected experts. That communication grows with batch size,
`num_experts_per_tok`, hidden dimension, and the number of expert-parallel
ranks. At high load, hot experts can receive many more tokens than cold
experts, so some devices become stragglers while others wait. In that regime,
the MoE can become limited by all-to-all bandwidth and token bucket imbalance,
making it more expensive and less predictable than a dense model with the same
training budget.
