torchrun --nproc_per_node=4  training_classifier.py \
    --model_name_or_path "google-t5/t5-base" \
    --data_path "./data/trajectory_set" \
    --bf16 True \
    --output_dir "./metric_learning_models/test_1114_gamma" \
    --run_name 'llm-fingerprint-1114_gamma' \
    --num_train_epochs 250 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    # --tf32 True