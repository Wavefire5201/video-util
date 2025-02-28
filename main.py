import cv2
import os
from datetime import timedelta

GO_PRO_PATH = r"C:\Users\15410\Downloads\GOPRO"


def extract_frames_at_interval(video_path, output_folder, interval_seconds=60):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    # Calculate frame numbers to extract (every minute)
    frames_to_extract = []
    for time_sec in range(0, int(duration), interval_seconds):
        frame_number = int(time_sec * fps)
        frames_to_extract.append(frame_number)

    # Extract and save frames
    for i, frame_number in enumerate(frames_to_extract):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()

        if ret:
            # Format timestamp for filename
            timestamp = str(timedelta(seconds=frame_number / fps)).replace(":", "-")

            # Save the frame
            frame_filename = os.path.join(output_folder, f"frame_{i+1}_{timestamp}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame at {frame_number/fps:.2f} seconds: {frame_filename}")

    # Release the video capture object
    cap.release()
    print(
        f"Extracted {len(frames_to_extract)} frames at {interval_seconds}-second intervals"
    )


# Example usage
# video_file = "videos/GH016298.MP4"
# output_folder = "extracted_frames"
# extract_frames_at_interval(
#     video_file, output_folder, 60
# )  # Extract a frame every 60 seconds (1 minute)

videos = []
for filename in os.listdir(GO_PRO_PATH):
    print(filename)
    if filename.lower().endswith(".mp4"):
        videos.append(filename)
videos.sort()
print(videos)

for video in videos:
    video_path = os.path.join(GO_PRO_PATH, video)
    # store in folder without .mp4
    output_folder = os.path.join(GO_PRO_PATH, "extracted_frames", video[:-4])
    extract_frames_at_interval(video_path, output_folder, 60)

# ffprobe -v quiet -select_streams v:0  -show_entries stream_tags=creation_time -of default=noprint_wrappers=1:nokey=1 input.mp4
