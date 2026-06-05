"""Model training cost analysis for Part 2.

You will implement three functions:

  - `model_training_cost_analysis_llama(config_path)`
  - `model_training_cost_analysis_deepseek(config_path)`
  - `get_optimal_N_D_from_cost(cost_budget)`

Run from the command line:

  python model_training_cost_analysis.py --model_config llama3_8b_config.json
  python model_training_cost_analysis.py --model_config deepseek_v3_config.json
  python model_training_cost_analysis.py --training_budget 5000000
"""
import argparse
import json
import math


def model_training_cost_analysis_llama(model_config_path):
    """Analyze training cost of a dense Llama-style model.

    Returns:
        total_params:   total trainable parameter count (int)
        flops_layer_TF: forward FLOPs of a single transformer layer (TFLOPs)
        peak_memory_GB: peak forward memory of a single transformer layer (GB)

    See the Part 2.1 writeup for the sequence-length / batch convention.
    """
    with open(model_config_path, "r") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"] # expanded internal MLP dimension.
    num_layers = config.get("num_hidden_layers", config.get("num_layers"))
    vocab_size = config["vocab_size"]
    seq_len = config["max_position_embeddings"]

    num_heads = config["num_attention_heads"]
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
    kv_hidden_size = num_kv_heads * head_dim

    embedding_params = vocab_size * hidden_size
    # output projection count
    lm_head_params = 0 if config.get("tie_word_embeddings", False) else hidden_size * vocab_size

    attention_params_per_layer = (
        hidden_size * hidden_size * 2           # q_proj and o_proj
        + hidden_size * kv_hidden_size * 2      # k_proj and v_proj
    )

    mlp_params_per_layer = (
        hidden_size * intermediate_size         # gate_proj
        + hidden_size * intermediate_size       # up_proj
        + intermediate_size * hidden_size       # down_proj
    )

    norm_params_per_layer = 2 * hidden_size
    final_norm_params = hidden_size

    params_per_layer = (
        attention_params_per_layer
        + mlp_params_per_layer
        + norm_params_per_layer
    )

    total_params = (
        embedding_params
        + num_layers * params_per_layer
        + final_norm_params
        + lm_head_params
    )

    projection_flops = 2 * seq_len * hidden_size * (
        hidden_size + kv_hidden_size + kv_hidden_size + hidden_size
    )

    qk_flops = 2 * num_heads * seq_len * seq_len * head_dim
    atten_v_flops = 2 * num_heads * seq_len * seq_len * head_dim

    mlp_flops = 6 * seq_len * hidden_size * intermediate_size
    Tflops_total_per_layer = (
        projection_flops
        + qk_flops
        + atten_v_flops
        + mlp_flops
    ) / 1e12

    bytes_per_elem = 2

    layer_param_bytes = params_per_layer * bytes_per_elem
    attention_activation_elems = (
        seq_len * hidden_size              # layer input
        + seq_len * hidden_size            # q
        + seq_len * kv_hidden_size         # k
        + seq_len * kv_hidden_size         # v
        + num_heads * seq_len * seq_len    # attention scores
        + num_heads * seq_len * seq_len    # attention probabilities
        + seq_len * hidden_size            # attention output/context
    )
    mlp_activation_elems = (
        seq_len * hidden_size
        + seq_len * intermediate_size      # gate
        + seq_len * intermediate_size      # up
        + seq_len * intermediate_size      # activated gate
        + seq_len * hidden_size
    )

    peak_activation_bytes = max(
        attention_activation_elems,
        mlp_activation_elems
    ) * bytes_per_elem

    peak_mem_GB = (layer_param_bytes + peak_activation_bytes) / 1e9

    return int(total_params), Tflops_total_per_layer, peak_mem_GB


def model_training_cost_analysis_deepseek(model_config_path):
    """Analyze training cost of a DeepSeek-V3-style MoE model.

    Same return signature as the Llama version. See the Part 2.3 writeup
    for the MLA attention and the dense-vs-MoE layer breakdown.
    """
    with open(model_config_path, "r") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    moe_intermediate_size = config["moe_intermediate_size"]
    num_layers = config["num_hidden_layers"]
    first_dense_layers = config["first_k_dense_replace"]
    num_moe_layers = num_layers - first_dense_layers
    vocab_size = config["vocab_size"]
    seq_len = config["max_position_embeddings"]

    num_heads = config["num_attention_heads"]
    q_lora_rank = config["q_lora_rank"]
    kv_lora_rank = config["kv_lora_rank"]
    qk_nope_head_dim = config["qk_nope_head_dim"]
    qk_rope_head_dim = config["qk_rope_head_dim"]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    v_head_dim = config["v_head_dim"]

    n_routed_experts = config["n_routed_experts"]
    n_shared_experts = config["n_shared_experts"]
    num_experts_per_tok = config["num_experts_per_tok"]

    embedding_params = vocab_size * hidden_size
    lm_head_params = (
        0 if config.get("tie_word_embeddings", False)
        else vocab_size * hidden_size
    )

    mla_attention_params = (
        hidden_size * q_lora_rank
        + q_lora_rank
        + q_lora_rank * num_heads * q_head_dim
        + hidden_size * (kv_lora_rank + qk_rope_head_dim)
        + kv_lora_rank
        + kv_lora_rank * num_heads * (qk_nope_head_dim + v_head_dim)
        + num_heads * v_head_dim * hidden_size
    )

    dense_mlp_params = 3 * hidden_size * intermediate_size
    expert_params = 3 * hidden_size * moe_intermediate_size
    moe_mlp_params = (
        n_routed_experts * expert_params
        + n_shared_experts * expert_params
        + hidden_size * n_routed_experts
    )

    norm_params_per_layer = 2 * hidden_size
    final_norm_params = hidden_size

    dense_layer_params = (
        mla_attention_params
        + dense_mlp_params
        + norm_params_per_layer
    )
    moe_layer_params = (
        mla_attention_params
        + moe_mlp_params
        + norm_params_per_layer
    )

    total_params = (
        embedding_params
        + lm_head_params
        + first_dense_layers * dense_layer_params
        + num_moe_layers * moe_layer_params
        + final_norm_params
    )

    projection_flops = 2 * seq_len * mla_attention_params
    qk_flops = 2 * num_heads * seq_len * seq_len * q_head_dim
    attn_v_flops = 2 * num_heads * seq_len * seq_len * v_head_dim
    attention_flops = projection_flops + qk_flops + attn_v_flops

    dense_mlp_flops = 6 * seq_len * hidden_size * intermediate_size
    active_moe_mlp_flops = (
        6
        * seq_len
        * hidden_size
        * moe_intermediate_size
        * (num_experts_per_tok + n_shared_experts)
    )
    router_flops = 2 * seq_len * hidden_size * n_routed_experts

    dense_layer_flops = attention_flops + dense_mlp_flops
    moe_layer_flops = attention_flops + active_moe_mlp_flops + router_flops
    avg_layer_flops_TF = (
        first_dense_layers * dense_layer_flops
        + num_moe_layers * moe_layer_flops
    ) / num_layers / 1e12

    bytes_per_elem = 2
    attention_activation_elems = (
        seq_len * hidden_size
        + seq_len * q_lora_rank
        + seq_len * num_heads * q_head_dim
        + seq_len * (kv_lora_rank + qk_rope_head_dim)
        + seq_len * num_heads * (qk_nope_head_dim + v_head_dim)
        + num_heads * seq_len * seq_len
        + num_heads * seq_len * seq_len
        + seq_len * num_heads * v_head_dim
    )
    dense_activation_elems = (
        seq_len * hidden_size
        + 3 * seq_len * intermediate_size
        + seq_len * hidden_size
    )
    active_moe_activation_elems = (
        seq_len * hidden_size
        + 3
        * seq_len
        * moe_intermediate_size
        * (num_experts_per_tok + n_shared_experts)
        + seq_len * hidden_size
    )

    peak_layer_param_bytes = max(dense_layer_params, moe_layer_params) * bytes_per_elem
    peak_activation_bytes = max(
        attention_activation_elems,
        dense_activation_elems,
        active_moe_activation_elems,
    ) * bytes_per_elem
    peak_memory_GB = (peak_layer_param_bytes + peak_activation_bytes) / 1e9

    return int(total_params), avg_layer_flops_TF, peak_memory_GB


def get_optimal_N_D_from_cost(cost_budget):
    """Pick the GPU and (N, D) that minimize loss under a $ training budget.

    cost_budget: a monetary training budget (in dollars)
    Returns:
        N: optimal model parameter count (absolute number)
        D: optimal training token count (absolute number)
        training_budget_flops: effective total training FLOPs
        best_gpu: name of the selected GPU, one of {'H100', 'H200', 'B200'}

    See the Part 2.2 writeup for the scaling law, the GPU price / TFLOPs
    table, and the MFU assumption.
    """
    gpus = {
        "H100": {"cost_per_hour": 3.0, "peak_tflops": 989},
        "H200": {"cost_per_hour": 4.0, "peak_tflops": 989},
        "B200": {"cost_per_hour": 6.0, "peak_tflops": 2250}
    }

    # Model FLOPs Utilization. Achieve 40% of theoretical peak compute during real training.
    mfu = 0.40

    best_gpu = None
    training_budget_flops = -1

    for gpu, spec in gpus.items():
        gpu_hours = cost_budget / spec["cost_per_hour"]
        effective_flops = (
            gpu_hours * 3600 * spec["peak_tflops"] * 1e12 * mfu
        )

        if effective_flops > training_budget_flops:
            training_budget_flops = effective_flops
            best_gpu = gpu
    
    A = 406.4
    alpha = 0.34
    B = 410.7
    beta = 0.29

    compute_constant = training_budget_flops / 6.0

    N = (
        (alpha * A)
        / (beta * B)
        * (compute_constant ** beta)
    ) ** (1.0 / (alpha + beta))

    D = compute_constant / N

    return N, D, training_budget_flops, best_gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training cost analysis")
    parser.add_argument("--model_config", type=str, help="Path to model config")
    parser.add_argument("--training_budget", type=float, default=None,
                        help="Training budget in dollars")
    args = parser.parse_args()

    if args.model_config:
        if "deepseek" in args.model_config:
            num_parameters, num_flops, memory_cost = (
                model_training_cost_analysis_deepseek(args.model_config)
            )
        elif "llama" in args.model_config:
            num_parameters, num_flops, memory_cost = (
                model_training_cost_analysis_llama(args.model_config)
            )
        else:
            print("Unknown model type — name your config llama*.json or deepseek*.json")
            raise SystemExit(1)
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(
            args.training_budget
        )
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")
