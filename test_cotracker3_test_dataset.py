import torch
import cv2
import time
import os
import glob
import numpy as np
from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline
from cotracker.utils.visualizer import Visualizer
from torchvision.transforms import Resize

# Set Tensor Cores precision
torch.set_float32_matmul_precision('medium')

# Load finetuned model
model = CoTrackerThreeOnline(window_len=16).cuda()  # Match checkpoint window length
checkpoint_path = "./checkpoints/finetuned/cotracker_three_final.pth"
if not os.path.exists(checkpoint_path):
    print(f"Checkpoint {checkpoint_path} not found.")
    exit(1)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
model.eval()

# Visualization directory
output_dir = "./checkpoints/finetuned/inference"
os.makedirs(output_dir, exist_ok=True)
visualizer = Visualizer(save_dir=output_dir, pad_value=0)

# Test dataset
test_dir = "/home/imerse/cotracker/data/custom/test"
video_paths = sorted(glob.glob(os.path.join(test_dir, "*.mp4")))
resize = Resize((256, 256))  # Match training crop_size

for video_path in video_paths:
    print(f"Processing {video_path}")
    # Load video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if len(frames) < 32:  # Match sequence_len
        print(f"Video {video_path} too short: {len(frames)} frames")
        continue
    frames = frames[:32]  # Limit to sequence_len
    video_array = np.stack(frames).astype(np.float32)  # Ensure numpy.ndarray
    print(f"Video array type: {type(video_array)}, shape: {video_array.shape}")
    video = torch.from_numpy(video_array).permute(0, 3, 1, 2) / 255.0  # Normalize to [0,1]
    video = resize(video).cuda()
    video = video.unsqueeze(0)  # [1, T, C, H, W]

    # Generate query points (128 points, matching traj_per_sample)
    query_points = torch.rand(1, 128, 3).cuda()  # [B, N, 3] (t, y, x)

    # Measure inference time and inspect outputs
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        outputs = model(video, queries=query_points)
        print(f"Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"Output {i} shape: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
        pred_tracks, pred_visibility = outputs[:2]  # Assume first two are tracks and visibility
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    fps = len(frames) / elapsed_time
    print(f"Inference time: {elapsed_time:.3f} s for {len(frames)} frames, FPS: {fps:.2f}")

    # Visualize and save output
    filename = os.path.basename(video_path).replace(".mp4", "")
    visualizer.visualize(
        video=video.clone(),
        tracks=pred_tracks,
        visibility=pred_visibility,
        filename=f"test_{filename}",
        writer=None
    )
    print(f"Tracks shape: {pred_tracks.shape}, Visibility shape: {pred_visibility.shape}")
    print(f"Output video saved to {output_dir}/test_{filename}.mp4")
