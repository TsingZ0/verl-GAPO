## GAPO: Robust Advantage Estimation for Real-World Code LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2312.04992-b31b1b.svg)](https://arxiv.org/abs/2510.21830)
![Apache License 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

*Group-Adaptive Advantage Denoiser for GRPO, DAPO, etc.*

Reinforcement learning (RL) is widely used for post-training large language models (LLMs) in code editing, where group-relative methods like GRPO are popular for their critic-free, normalized advantage estimation. However, in real-world code-editing scenarios, reward distributions are often skewed with unpredictable ***noisy outliers***, leading to distorted advantage computation and increased noise. To address this issue, we propose Group Adaptive Policy Optimization (GAPO), which adaptively find an outlier-free highest-density interval (HDI) per prompt and then uses the median of that interval as an adaptive Q to replace the group mean in advantage calculation. This adaptive Q robustly handles skewed distributions while remaining plug-and-play and efficient. 

### How to Use

```bash
# Configure MODEL_PATH, CKPTS_DIR, TRAIN_FILE, TEST_FILE in run_grpo.py and run_dapo.py before you run.

python -u run_grpo.py --reward_function edem --model_name Qwen2.5-Coder-7B-Instruct --GPUs 0,1,2,3,4,5,6,7 --rollout_bsz 512 --update_bsz 32 --rollout_n 8 --find_method median --verbose median-div # for grpo
python -u run_dapo.py --reward_function edem --model_name Qwen2.5-Coder-7B-Instruct --GPUs 0,1,2,3,4,5,6,7 --rollout_bsz 512 --update_bsz 32 --rollout_n 8 --find_method median --verbose median-div # for dapo
```