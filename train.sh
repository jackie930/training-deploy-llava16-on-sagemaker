accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir sft-llava-1.6-7b-hf \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --num_train_epochs 1 