import argparse
import os
import subprocess
import socket

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="DAPO-Code-Edit")
    parser.add_argument("--reward_function", type=str, default="ed")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--GPUs", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--rollout_bsz", type=int, default=512)
    parser.add_argument("--update_bsz", type=int, default=32)
    parser.add_argument("--rollout_n", type=int, default=16)
    parser.add_argument("--gen_x", type=int, default=3)
    parser.add_argument("--verbose", type=str, default="")
    parser.add_argument("--adv_estimator", type=str, default="grpo")
    parser.add_argument("--find_method", type=str, default=None)
    parser.add_argument("--find_target_fraction", type=float, default=0.5)
    parser.add_argument("--save_folder", type=str, default="saved_contents")
    args = parser.parse_args()

    # register reward function
    if args.reward_function == "em":
        reward_function = "compute_score_em"
    elif args.reward_function == "ed":
        reward_function = "compute_score_ed"
    elif args.reward_function == "edem":
        reward_function = "compute_score_edem"
    elif args.reward_function == "bleu":
        reward_function = "compute_score_bleu"
    else:
        raise ValueError(f"Unknown reward function: {args.reward_function}")
    model_name = args.model_name
    GPUs = args.GPUs

    save_contents = "['hf_model']"
    project_name = args.project_name + f"-{args.reward_function}"
    exp_name = f"{model_name}-{args.reward_function}"
    save_folder = args.save_folder
    if args.verbose:
        exp_name += f"-{args.verbose}"
    n_gpus_per_node = len(GPUs.split(","))

    adv_estimator = args.adv_estimator
    use_kl_in_reward = "False"
    kl_coef = 0.0
    use_kl_loss = "False"
    kl_loss_coef = 0.0
    clip_ratio_low = 0.2
    clip_ratio_high = 0.28

    max_prompt_length = 1024 * 4
    max_response_length = 1024 * 1
    enable_overlong_buffer = "True"
    overlong_buffer_len = 1024 * 1
    overlong_penalty_factor = 1.0

    loss_agg_mode = "token-mean"
    enable_filter_groups = "True"
    filter_groups_metric = "score"
    max_num_gen_batches = 0
    train_prompt_bsz = args.rollout_bsz
    gen_prompt_bsz = args.rollout_bsz * args.gen_x
    n_resp_per_prompt = args.rollout_n
    train_prompt_mini_bsz = args.update_bsz

    NNODES = 1
    MODEL_PATH = f"/YOUR_DATA_FOLDER/models/{model_name}"
    CKPTS_DIR = f"/YOUR_DATA_FOLDER/verl-GAPO/models/{project_name}/{exp_name}"
    TRAIN_FILE = "/YOUR_DATA_FOLDER/trainset.parquet"
    TEST_FILE = "/YOUR_DATA_FOLDER/testset.parquet"

    temperature = 1.0
    top_p = 1.0
    top_k = -1
    val_n = 1

    sp_size = 4
    gen_tp = 4
    use_dynamic_bsz = "True"
    actor_ppo_max_token_len = max_prompt_length + max_response_length
    infer_ppo_max_token_len = max_prompt_length + max_response_length
    offload = "True"

    total_epochs = 10
    test_freq = 1
    save_freq = 10000000

    # new method
    find_method = args.find_method
    find_target_fraction = args.find_target_fraction

    cmd = [
        "python3", "-m", "recipe.dapo.main_dapo",
        f"data.train_files={TRAIN_FILE}",
        f"data.val_files={TEST_FILE}",
        "data.prompt_key=prompt",
        "data.truncation=left",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        f"data.gen_batch_size={gen_prompt_bsz}",
        f"data.train_batch_size={train_prompt_bsz}",
        f"algorithm.adv_estimator={adv_estimator}",
        f"algorithm.use_kl_in_reward={use_kl_in_reward}",
        f"algorithm.kl_ctrl.kl_coef={kl_coef}",
        f"algorithm.filter_groups.enable={enable_filter_groups}",
        f"algorithm.filter_groups.max_num_gen_batches={max_num_gen_batches}",
        f"algorithm.filter_groups.metric={filter_groups_metric}",
        f"+algorithm.find_method={find_method}",
        f"+algorithm.find_target_fraction={find_target_fraction}",
        f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
        f"actor_rollout_ref.actor.use_kl_loss={use_kl_loss}",
        f"actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}",
        f"actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}",
        f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}",
        "actor_rollout_ref.actor.clip_ratio_c=10.0",
        "actor_rollout_ref.model.use_remove_padding=True",
        f"actor_rollout_ref.actor.use_dynamic_bsz={use_dynamic_bsz}",
        f"actor_rollout_ref.ref.log_prob_use_dynamic_bsz={use_dynamic_bsz}",
        f"actor_rollout_ref.rollout.log_prob_use_dynamic_bsz={use_dynamic_bsz}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={actor_ppo_max_token_len}",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={infer_ppo_max_token_len}",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={infer_ppo_max_token_len}",
        f"actor_rollout_ref.model.path={MODEL_PATH}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=False",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
        "actor_rollout_ref.actor.optim.weight_decay=0.1",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.grad_clip=1.0",
        f"actor_rollout_ref.actor.loss_agg_mode={loss_agg_mode}",
        f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={sp_size}",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.80",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={gen_tp}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={max_prompt_length + max_response_length}",
        f"actor_rollout_ref.rollout.temperature={temperature}",
        f"actor_rollout_ref.rollout.top_p={top_p}",
        f"actor_rollout_ref.rollout.top_k={top_k}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={temperature}",
        f"actor_rollout_ref.rollout.val_kwargs.top_k={top_k}",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        f"actor_rollout_ref.rollout.val_kwargs.n={val_n}",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={offload}",
        f"actor_rollout_ref.ref.ulysses_sequence_parallel_size={sp_size}",
        "actor_rollout_ref.actor.fsdp_config.fsdp_size=-1",
        f"actor_rollout_ref.actor.checkpoint.save_contents={save_contents}",
        "reward_model.reward_manager=dapo",
        f"reward_model.overlong_buffer.enable={enable_overlong_buffer}",
        f"reward_model.overlong_buffer.len={overlong_buffer_len}",
        f"reward_model.overlong_buffer.penalty_factor={overlong_penalty_factor}",
        "trainer.logger=[\"console\",\"wandb\"]",
        f"trainer.project_name={project_name}",
        f"trainer.experiment_name={exp_name}",
        f"trainer.nnodes={NNODES}",
        f"trainer.n_gpus_per_node={n_gpus_per_node}",
        "trainer.val_before_train=True",
        f"trainer.test_freq={test_freq}",
        f"trainer.save_freq={save_freq}",
        f"trainer.total_epochs={total_epochs}",
        f"trainer.default_local_dir={CKPTS_DIR}",
        "trainer.resume_mode=disable",
        "+trainer.remove_previous_ckpt_in_save=True",
        f"+trainer.save_folder={save_folder}",
        "custom_reward_function.path=code_edit_reward.py",
        f"custom_reward_function.name={reward_function}",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPUs

    subprocess.run(cmd, env=env, check=True)

if __name__ == "__main__":
    # Print current machine's real IP address
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable
        s.connect(('8.8.8.8', 80))
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = '127.0.0.1'
    finally:
        s.close()
    print(f"Current machine's IP address: {ip_address}")

    main()