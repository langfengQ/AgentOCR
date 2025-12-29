set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3

###############
# Highlight configs: use environment variable to avoid Hydra parsing issues with < > characters
# Format: "context1:r,g,b;context2:r,g,b"
# <search> and </search> are highlighted in blue (0,0,255)
# <information> and </information> are highlighted in red (255,0,0)
###############
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

use_ocr=True
ocr_use_parallel=True
ocr_max_workers=64

ocr_font_size=12
ocr_max_width=560
ocr_max_height=4096

# Compact mode settings (replace newlines with colored symbols)
compact_mode_enable=False

# Agent-selected compression settings
agent_select_compression_enable=True
compression_reward_coef=0.01  # base coefficient for compression reward
compression_failure_penalty_coef=0.0 # >0 to enable compression-based penalty on failed trajectories
compression_reward_every_n_steps=3  # apply compression reward every n steps

train_data_size=128
val_data_size=512
group_size=8
# Set mode based on use_ocr: visual if use_ocr=True, text otherwise
if [ "$use_ocr" = "True" ]; then
    max_prompt_length=4096
    model=Qwen/Qwen2.5-VL-3B-Instruct
    experiment_name="ocr_compact${compact_mode_enable}_selfcompress${agent_select_compression_enable}_maxprompt${max_prompt_length}_poscoef${compression_reward_coef}_negcoef${compression_failure_penalty_coef}_everyn${compression_reward_every_n_steps}_fs${ocr_font_size}_maxwidth${ocr_max_width}_maxheight${ocr_max_height}_qwen2.5_vl_3b"
else
    max_prompt_length=14000
    model=Qwen/Qwen2.5-3B-Instruct
    experiment_name="text_maxprompt${max_prompt_length}_qwen2.5_3b"
fi

TRAIN_DATA="$HOME/data/searchR1_processed_direct/train.parquet"
VAL_DATA="$HOME/data/searchR1_processed_direct/test.parquet"


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.search_url='http://127.0.0.1:7856/retrieve' \
    env.search.topk=3 \
    ocr.use_ocr=$use_ocr \
    ocr.use_parallel=$ocr_use_parallel \
    ocr.max_workers=$ocr_max_workers \
    ocr.font_size=$ocr_font_size \
    ocr.max_width=$ocr_max_width \
    ocr.max_height=$ocr_max_height \
    ocr.compact_mode.enable=$compact_mode_enable \
    ocr.agent_select_compression.enable=$agent_select_compression_enable \
    ocr.agent_select_compression.compression_reward_coef=$compression_reward_coef \
    ocr.agent_select_compression.compression_failure_penalty_coef=$compression_failure_penalty_coef \
    ocr.agent_select_compression.compression_reward_every_n_steps=$compression_reward_every_n_steps \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='AgentOCR_search' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False $@

