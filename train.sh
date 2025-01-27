torchrun --nproc_per_node=4  finetuning.py \
    --model_name_or_path "/mnt/data/yuliangyan/microsoft/Phi-3-medium-4k-instruct" \
    --data_path "tatsu-lab/alpaca" \
    --bf16 True \
    --output_dir "./instruction_tuning_models/phi3-instruct-ft" \
    --lora_rank 4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True