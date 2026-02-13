export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch \
  --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path data/train_wan \
  --dataset_metadata_path data/train_wan/metadata.csv \
  --height 1280 \
  --width 704 \
  --num_frames 121 \
  --trainable_models "dit" \
  --model_paths '[
    [
        "/data/models/text-or-image-to-video-models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
        "/data/models/text-or-image-to-video-models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
        "/data/models/text-or-image-to-video-models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"
    ],
    "/data/models/text-or-image-to-video-models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
    "/data/models/text-or-image-to-video-models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"
  ]' \
  --learning_rate 5e-6 \
  --dataset_repeat 6 \
  --num_epochs 3 \
  --output_path "./models/train/Wan2.2-TI2V-5B_sla_sft" \
  --extra_inputs "input_image" \
  --task "sla_sft"
