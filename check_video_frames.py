import cv2
import os
import glob

def get_frame_count(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, f"Failed to open {video_path}"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count, None
    except Exception as e:
        return None, f"Error processing {video_path}: {str(e)}"

def check_videos(directory, min_frames=32):
    video_paths = sorted(glob.glob(os.path.join(directory, "*.mp4")))
    print(f"Checking videos in {directory}...")
    for video_path in video_paths:
        frame_count, error = get_frame_count(video_path)
        if error:
            print(f"{video_path}: {error}")
        else:
            status = "OK" if frame_count >= min_frames else f"Too short (<{min_frames} frames)"
            print(f"{video_path}: {frame_count} frames, {status}")

if __name__ == "__main__":
    train_dir = "./data/custom/train"
    test_dir = "./data/custom/test"
    eval_dir = "./data/custom/eval"
    check_videos(train_dir, min_frames=32)
    check_videos(test_dir, min_frames=32)
    check_videos(eval_dir, min_frames=32)
