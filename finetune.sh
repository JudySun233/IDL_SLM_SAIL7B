# re‑login wandb if you need to
wandb login

accelerate launch \
  --mixed_precision bf16 \
  --deepspeed_config_file ds_config.json \
  finetune.py \
    --data_path               datasets/SAIL_train_deberta_entailment.json \
    --run_name                "test_1" \
    --output_dir              ./llama2-lora-entail \
    --lora_r                  16 \
    --lora_alpha              32 \
    --lora_dropout            0.1 \
    --qlora \
    --num_train_epochs        1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size  1 \
    --gradient_accumulation_steps 16 \
    --learning_rate           2e-5 \
    --warmup_ratio            0.03 \
    --lr_scheduler_type       cosine \
    --save_steps              400 \
    --save_total_limit        10 \
    --model_max_length        512
