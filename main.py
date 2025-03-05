import os
import json
from src.extract_weight import extract_weight_gemini
from src.frame_utilities import extract_frames_at_interval

# GO_PRO_PATH = "/run/media/wavefire/6565-3263/DCIM/100GOPRO/"
GO_PRO_PATH = r"D:\DCIM\100GOPRO"
FRAMES_FOLDER = "modified_frames"
RESULT_FILE_NAME = "3-3_to_3-4_Model_2015.json"

videos = sorted(os.listdir(GO_PRO_PATH))
print(
    f"Number of MP4s found: {sum(1 for video in videos if video.lower().endswith('.mp4'))}"
)

os.makedirs(FRAMES_FOLDER, exist_ok=True)
for filename in sorted(os.listdir(GO_PRO_PATH)):
    if filename.lower().endswith(".mp4"):
        video_path = os.path.join(GO_PRO_PATH, filename)
        output_folder = os.path.join(FRAMES_FOLDER, filename[:-4])
        if os.path.exists(output_folder):
            print(f"Skipping {filename}, already exists")
            continue
        extract_frames_at_interval(video_path, output_folder, 60)


# Load existing results if file exists
try:
    with open(RESULT_FILE_NAME, "r") as f:
        results = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    results = []

total_frames_extracted = 0

for folder in sorted(os.listdir(FRAMES_FOLDER)):
    folder_path = os.path.join(FRAMES_FOLDER, folder)
    if os.path.isdir(folder_path):
        print("Extracting weights from:", folder)
        frames_in_folder = 0

        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".jpg"):
                image_date = filename[:-4]  # Remove .jpg extension

                # Check if this date already exists in results
                if any(d["date"] == image_date for d in results):
                    print(f"Skipping {filename}, already exists")
                else:
                    previous_weight = results[-1]["weight"] if results else None
                    weight = extract_weight_gemini(
                        os.path.join(folder_path, filename), previous_weight
                    )
                    results.append({"date": image_date, "weight": weight})
                    print(f"{image_date}: {weight}")

                frames_in_folder += 1

        total_frames_extracted += frames_in_folder
        print(f"Total frames extracted from {folder}: {frames_in_folder}")

        # Save after processing each folder (as a checkpoint)
        with open(RESULT_FILE_NAME, "w") as f:
            json.dump(results, f)

print(f"Total frames extracted: {total_frames_extracted}")

# Final save at the end
with open(RESULT_FILE_NAME, "w") as f:
    json.dump(results, f)
