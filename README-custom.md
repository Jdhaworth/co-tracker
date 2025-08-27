# Custom CoTracker Setup for NVIDIA IGX Orin Development Kit

This fork of the CoTracker repository[](https://github.com/facebookresearch/co-tracker) includes modifications to finetune and test the CoTracker3 model on a custom dataset using the NVIDIA IGX Orin Development Kit (Jetson Orin with RTX 6000 Ada GPU, CUDA 12.6, driver 570.181).

## Hardware and Environment
- **System**: NVIDIA IGX Orin Development Kit
  - GPU: NVIDIA RTX 6000 Ada (48 GB VRAM, 182 TFLOPS FP16/BF16)
  - CPU: 12-core Arm Cortex-A78AE (~2.2 GHz)
  - RAM: ≥32 GB LPDDR5
  - OS: Ubuntu 22.04 (JetPack 6.1)
- **CUDA**: 12.6
- **Driver**: 570.181

## Installation Instructions
1. **Create Conda Environment**:
   \`\`\`bash
   conda create -n cotracker python=3.10
   conda activate cotracker
   \`\`\`

2. **Install PyTorch and torchvision** (custom wheels for Jetson aarch64):
   \`\`\`bash
   wget https://nvidia.box.com/shared/static/mv8550r5xoth3u9xs4vps3r4hbt0g7yg.whl -O torch-2.3.0-cp310-cp310-linux_aarch64.whl
   pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl
   wget https://nvidia.box.com/shared/static/1z7sm1u6xtv38vmx4h29q5ypxh1k7fvs.whl -O torchvision-0.18.0-cp310-cp310-linux_aarch64.whl
   pip install torchvision-0.18.0-cp310-cp310-linux_aarch64.whl
   \`\`\`
   Note: Replace URLs with your saved wheel files if stored locally.

3. **Install Dependencies**:
   \`\`\`bash
   pip install opencv-python==4.12.0.88
   pip install numpy==1.26.0  # Downgraded to fix torch.from_numpy bug
   conda install -c conda-forge moviepy==1.0.3 imageio==2.37.0 imageio-ffmpeg==0.6.0 requests==2.32.5
   sudo apt-get update && sudo apt-get install -y ffmpeg
   pip install tqdm==4.67.1 pillow==11.3.0 python-dotenv==1.1.1 decorator==5.2.1 proglog==0.1.12
   pip install lightning==2.0.0  # Compatible with Fabric
   \`\`\`

4. **Clone and Setup Fork**:
   \`\`\`bash
   git clone https://github.com/<your-username>/co-tracker.git ~/cotracker-fork
   cd ~/cotracker-fork
   \`\`\`

## Modified Scripts
- **train_on_real_data.py**: Modified for `CustomDataset`, absolute paths, `SingleDeviceStrategy`, and correct argument handling (e.g., `self.save(save_path, save_dict)`).
- **train_utils.py**: Updated `run_test_eval` to use `model` directly, avoiding `model.module.module`.
- **utils.py**: Added `valid` field to `CoTrackerData` and updated `collate_fn` to stack `valid`.
- **custom_dataset.py**: Added resizing (256x256), `valid` tensor, and dummy `trajectory`/`visibility` for training.
- **check_video_frames.py**: Script to verify video frame counts.
- **test_cotracker3_test_dataset.py**: Inference script for testing on custom test dataset, measuring FPS, and visualizing tracks.

## Dataset Structure
- **Training**: `/home/imerse/cotracker/data/custom/train/*.mp4` (11 videos, ≥32 frames)
- **Test**: `/home/imerse/cotracker/data/custom/test/*.mp4` (3 videos)
- **Eval**: `/home/imerse/cotracker/data/custom/eval/*.mp4` (3 videos)
- **Evaluation Dataset**: `/home/imerse/cotracker/data/tapvid/tapvid_davis/tapvid_davis.pkl`

## Finetuning Command
\`\`\`bash
python train_on_real_data.py \
  --batch_size 1 \
  --num_steps 5000 \
  --ckpt_path ./checkpoints/finetuned \
  --model_name cotracker_three \
  --save_freq 200 \
  --sequence_len 32 \
  --eval_datasets tapvid_davis_first \
  --traj_per_sample 128 \
  --save_every_n_epoch 15 \
  --evaluate_every_n_epoch 15 \
  --model_stride 4 \
  --dataset_root ./data/custom \
  --num_nodes 1 \
  --real_data_splits 0 \
  --num_virtual_tracks 64 \
  --mixed_precision \
  --random_frame_rate \
  --restore_ckpt ./checkpoints/baseline_online.pth \
  --lr 0.00005 \
  --validate_at_start \
  --sliding_window_len 32 \
  --limit_samples 5000 \
  --crop_size 256 256
\`\`\`

## Inference Command
\`\`\`bash
python ~/test_cotracker3_test_dataset.py
\`\`\`

## Notes
- Use `tensorboard --logdir ./checkpoints/finetuned` to monitor training metrics.
- Check GPU usage with `nvidia-smi` (~28 GB VRAM during training, ~2–5 GB during inference).
- Downgraded NumPy to 1.26.0 to fix `torch.from_numpy` bug on aarch64.
- Videos must have ≥32 frames (use `check_video_frames.py` to verify).

## Custom Wheels
Store custom PyTorch and torchvision wheels locally:
- `torch-2.3.0-cp310-cp310-linux_aarch64.whl`
- `torchvision-0.18.0-cp310-cp310-linux_aarch64.whl`
Place in a shared directory (e.g., `/home/imerse/wheels/`) and update installation commands accordingly.

