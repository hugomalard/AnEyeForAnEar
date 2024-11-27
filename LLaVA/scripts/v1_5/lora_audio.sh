#!/bin/bash

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path /home/ids/hmalard/audible/protected/dev/dcase/studies/013_Llava_Mistral_Caption/badAlign.json \
    --image_folder /home/ids/hmalard \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --audio_tower laion/clap-htsat-unfused \
    --pretrain_mm_mlp_adapter /home/ids/hmalard/testL/LLaVA/checkpoints/audio-vis-soundScene-llava-v1.5-7b/audio_projector_audio_vis.bin \
    --mm_projector_type mlp2x_gelu \
    --modality audio_vis \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/lora/bad_align_audio_vis_mask_llava-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb