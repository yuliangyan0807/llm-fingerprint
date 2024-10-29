torchrun --nproc_per_node=4  dpo_finetuning.py \
    --model_name_or_path "/home/yuliangyan/Code/llm-fingerprinting/instruction_tuning_models/llama3-ft" \
    --data_path "Intel/orca_dpo_pairs" \
    --bf16 True \
    --output_dir "./dpo_tuning_models/llama3-ft-dpo" \
    --lora_rank 4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_grad_norm 0.3 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --tf32 True