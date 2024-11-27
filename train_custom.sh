accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm.py \
    --dataset_name customer \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir sft-llava-1.6-7b-hf-customer2batch \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --num_train_epochs 20 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --per_device_eval_batch_size 8 