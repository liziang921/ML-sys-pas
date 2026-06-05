# Part 3 — Speculative Decoding Report

## Setup

I implemented single-batch greedy speculative decoding with
`EleutherAI/pythia-1.4b-deduped` as the target model and
`EleutherAI/pythia-160m-deduped` as the draft model. The two models share the
same tokenizer, which makes token-by-token verification straightforward. Each
benchmark generated 100 new tokens per prompt and averaged over three prompts
and three runs per prompt.

The decoder uses greedy generation (`do_sample=False`) for both target-only and
speculative decoding. Models were loaded in fp16 on CUDA, inference ran under
`torch.inference_mode()`, and CUDA synchronization was used around timing
regions so the reported wall-clock times reflect completed GPU work.

## Sweep Results

| num speculative tokens | acceptance rate | speedup | speculative tok/s | baseline tok/s |
|---:|---:|---:|---:|---:|
| 2 | 99.02% | 1.21x | 149.90 | 123.83 |
| 4 | 97.19% | 1.79x | 162.68 | 111.17 |
| 8 | 95.12% | 1.48x | 187.48 | 126.35 |
| 16 | 88.22% | 1.19x | 158.16 | 117.74 |

All four settings exceeded the 75% acceptance threshold, and all four achieved
at least 1.0x speedup over target-only decoding. The best measured speedup was
at `k=4` with 1.79x, while `k=8` gave the highest speculative throughput at
187.48 tokens/sec.

## Analysis

Acceptance decreased as the speculative chain length increased. At `k=2`, the
draft tokens were accepted 99.02% of the time. At `k=16`, acceptance fell to
88.22%. This is expected: longer draft chains have more chances for the draft
model and target model to disagree, and after the first mismatch the remaining
draft suffix is discarded.

Speedup depends on both acceptance and overhead. With `k=2`, acceptance was
very high, but the decoder needed many target verification rounds, so speedup
was only 1.21x. Increasing to `k=4` reduced the number of target calls while
still keeping 97.19% acceptance, giving the best speedup. `k=8` had slightly
lower speedup in this run but the highest tokens/sec, showing that it was also
a strong setting. At `k=16`, the lower acceptance rate and one slow outlier run
reduced the average speedup.

Overall, `k=4` and `k=8` were the best tradeoffs in this experiment. Small `k`
keeps acceptance high but does not amortize verification enough; large `k`
reduces verifier calls but wastes more draft work after mismatches.

## Optimization Ablation

I also tested a draft-model KV-cache toggle at `k=8`. I repeated each setting
three times and report the mean below.

| setting | acceptance rate | speedup | speculative tok/s | baseline tok/s |
|---|---:|---:|---:|---:|
| fp16 + draft KV cache | 95.12% | 1.51x | 171.28 | 116.77 |
| fp16 + no draft KV cache | 95.12% | 1.64x | 182.82 | 119.29 |

The acceptance rate was identical, which is expected because KV caching changes
computation reuse, not greedy token choices. In this short-sequence benchmark,
disabling the draft KV cache did not hurt performance and was slightly faster
on average. I interpret this as a workload-specific result rather than a
general rule: the draft model is much smaller than the target model, each
speculative proposal is only eight tokens, and target verification dominates
the cost. The repeated runs also showed noticeable timing variance, including
occasional slow outliers. For the main implementation I kept KV caching enabled
because it is the standard inference setting and should help more for longer
draft chains or larger draft models.

Greedy decoding was important for stable acceptance measurements, since
sampling would introduce random draft choices and lower deterministic agreement
with the target. fp16 reduced memory use and improved GPU throughput compared
with full precision while preserving the greedy outputs for this benchmark.


## Bonus: N-Gram Lookup Decoding

I implemented n-gram lookup decoding as a lightweight speculative source before
the normal draft model. At each step, the decoder searches the generated token
context for a repeated recent suffix. If it finds a previous occurrence, it
copies the tokens that followed that occurrence and uses them as speculative
tokens; otherwise it falls back to the Pythia draft model. The target model
still verifies every proposed token, so incorrect n-gram guesses are rejected.

Using the same three prompts as the main benchmark at `k=8`, the n-gram variant
achieved 95.12% acceptance, 2.59x speedup, 401.41 speculative tok/s, and
115.57 baseline tok/s, with an n-gram hit rate of 84.13%. The acceptance rate
matched the standard `k=8` run, but throughput was much higher because many
draft proposals came from cheap token lookup instead of a neural draft-model
forward pass. N-gram lookup is most useful when generated text repeats phrases,
templates, code patterns, or lyrics. It is less general than learned tree
speculation methods, because copied continuations can be wrong in novel prose,
but target verification keeps the final output correct by rejecting mismatched
tokens.