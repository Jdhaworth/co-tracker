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
   ```bash
   conda create -n cotracker python=3.10
   conda activate cotracker
   ```

2. **Install PyTorch and torchvision** (download wheels from Google Drive):
   ```bash
   wget -O torch-2.3.0-cp310-cp310-linux_aarch64.whl 'https://drive.google.com/uc?export=download&id=1cqjycU2R9TzcsVmT26T8BKC0WR1eCRx3'
   wget -O torchvision-0.18.0-cp310-cp310-linux_aarch64.whl 'https://drive.google.com/uc?export=download&id=1oRxTsA_Y98kZadewUHRCu0MN0q9HmXSI'
   pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl
   pip install torchvision-0.18.0-cp310-cp310-linux_aarch64.whl
   ```
   Alternatively, place wheels in ./wheels/ and install:
   ```bash
   pip install ./wheels/torch-2.3.0-cp310-cp310-linux_aarch64.whl
   pip install ./wheels/torchvision-0.18.0-cp310-cp310-linux_aarch64.whl
   ```

3. **Install Dependencies**:
   ```bash
   pip install opencv-python==4.12.0.88 numpy==1.26.0
   conda install -c conda-forge moviepy==1.0.3 imageio==2.37.0 imageio-ffmpeg==0.6.0 requests==2.32.5
   sudo apt-get update && sudo apt-get install -y ffmpeg
   pip install tqdm==4.67.1 pillow==11.3.0 python-dotenv==1.1.1 decorator==5.2.1 proglog==0.1.12
   pip install lightning==2.0.0
   ```

4. **Clone and Setup Fork**:
   ```bash
   git clone https://github.com/<your-username>/co-tracker.git
   cd co-tracker
   ```

## Dataset Structure
- **Training**: ./data/custom/train/*.mp4 (minimum 32 frames)
- **Test**: ./data/custom/test/*.mp4 (minimum 32 frames)
- **Eval**: ./data/custom/eval/*.mp4 (minimum 32 frames)
- **Evaluation Dataset**: ./data/tapvid/tapvid_davis/tapvid_davis.pkl
Place your MP4 videos and tapvid_davis.pkl in the respective directories. Use check_video_frames.py to verify frame counts.

## Modified Scripts
- **train_on_real_data.py**: Uses relative paths, CustomDataset, SingleDeviceStrategy, fixed checkpoint saving.
- **train_utils.py**: Updated run_test_eval for model directly.
- **utils.py**: Added valid field to CoTrackerData, updated collate_fn.
- **custom_dataset.py**: Added resizing (256x256), valid tensor, dummy trajectory/visibility.
- **check_video_frames.py**: Verifies video frame counts for train/test/eval.
- **test_cotracker3_test_dataset.py**: Tests finetuned model, measures FPS, visualizes tracks.

## Finetuning Command
```bash
python train_on_real_data.py \\
  --batch_size 1 \\
  --num_steps 5000 \\
  --ckpt_path ./checkpoints/finetuned \\
  --model_name cotracker_three \\
  --save_freq 200 \\
  --sequence_len 32 \\
  --eval_datasets tapvid_davis_first \\
  --traj_per_sample 128 \\
  --save_every_n_epoch 15 \\
  --evaluate_every_n_epoch 15 \\
  --model_stride 4 \\
  --dataset_root ./data/custom \\
  --num_nodes 1 \\
  --real_data_splits 0 \\
  --num_virtual_tracks 64 \\
  --mixed_precision \\
  --random_frame_rate \\
  --restore_ckpt ./checkpoints/baseline_online.pth \\
  --lr 0.00005 \\
  --validate_at_start \\
  --sliding_window_len 32 \\
  --limit_samples 5000 \\
  --crop_size 256 256
```

## Inference Command
```bash
python test_cotracker3_test_dataset.py
```

## Notes
- Use `tensorboard --logdir ./checkpoints/finetuned` to monitor metrics.
- Check GPU usage with `nvidia-smi` (~28 GB VRAM during training, ~2–5 GB during inference).
- NumPy 1.26.0 fixes `torch.from_numpy` bug on aarch64.
- Videos must have ≥32 frames (use check_video_frames.py to verify).
- Wheels are hosted on Google Drive due to size; download from provided links.
