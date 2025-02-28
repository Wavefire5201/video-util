import cv2
import os
import datetime
import subprocess
import json
from extract_weight import extract_weight_gemini

GO_PRO_PATH = "/home/wavefire/Downloads/GOPRO"

# Coordinates and dimensions for the cropped region
x = 577
y = 528
width = 815
height = 365


def extract_date(video_path) -> float:
    # ffprobe -v quiet -select_streams v:0  -show_entries stream_tags=creation_time -of default=noprint_wrappers=1:nokey=1 input.mp4
    command = [
        "ffprobe",
        "-v",
        "quiet",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream_tags=creation_time",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    creation_time = result.stdout.strip()
    dt = datetime.datetime.fromisoformat(creation_time)
    print(f"{video_path}: {dt}")
    print(dt.timestamp())
    return dt.timestamp()


# extract_date("/home/wavefire/Downloads/GOPRO/GH016298.MP4")


def extract_frames_at_interval(video_path, output_folder, interval_seconds=60):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    date = extract_date(video_path)
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
            # Crop the frame to the specified region
            cropped_frame = frame[y : y + height, x : x + width]
            gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            bw_frame = cv2.adaptiveThreshold(
                gray_frame,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            )

            # Format timestamp for filename
            timestamp = str(datetime.timedelta(seconds=frame_number / fps)).replace(
                ":", "-"
            )

            # Save the cropped frame
            # frame_filename = os.path.join(output_folder, f"frame_{i+1}_{timestamp}.jpg")
            frame_filename = os.path.join(
                output_folder, f"{date + (frame_number / fps)}.jpg"
            )
            cv2.imwrite(frame_filename, gray_frame)
            print(f"Saved frame at {frame_number/fps:.2f} seconds: {frame_filename}")

    # Release the video capture object
    cap.release()
    print(
        f"Extracted {len(frames_to_extract)} frames at {interval_seconds}-second intervals"
    )


# video_file = "/home/wavefire/Downloads/GOPRO/GH016298.MP4"
# output_folder = "extracted_frames"
# extract_frames_at_interval(
#     video_file, output_folder, 60
# )  # Extract a frame every 60 seconds (1 minute)

# for filename in sorted(os.listdir(GO_PRO_PATH)):
#     print(filename)
#     if filename.lower().endswith(".mp4"):
#         video_path = os.path.join(GO_PRO_PATH, filename)
#         output_folder = os.path.join(GO_PRO_PATH, "extracted_frames", filename[:-4])
#         extract_frames_at_interval(video_path, output_folder, 60)

results = []
for folder in sorted(os.listdir(os.path.join(GO_PRO_PATH, "extracted_frames"))):
    if os.path.isdir(os.path.join(GO_PRO_PATH, "extracted_frames", folder)):
        print("Extracting weights from: ", folder)
        for filename in sorted(
            os.listdir(os.path.join(GO_PRO_PATH, "extracted_frames", folder))
        ):
            if filename.endswith(".jpg"):
                weight = extract_weight_gemini(
                    os.path.join(GO_PRO_PATH, "extracted_frames", folder, filename)
                )
                results.append({"date": filename[:-4], "weight": weight})
                print(filename[:-4], weight)
with open("results.json", "w") as f:
    json.dump(results, f)
