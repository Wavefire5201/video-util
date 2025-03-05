import datetime
import os
import subprocess

import cv2
import numpy as np

# Coordinates and dimensions for the cropped region
x = 577
y = 528
width = 815
height = 365


def extract_date(video_path: str) -> float:
    """
    Extract the creation timestamp from a video file using ffprobe.

    This function uses ffprobe to extract the creation_time metadata from the
    video stream of the provided file. The extracted timestamp is converted to
    a Unix timestamp (seconds since epoch).

    Args:
        video_path: Path to the video file from which to extract the creation date

    Returns:
        float: Unix timestamp (seconds since epoch) of the video's creation time

    Raises:
        subprocess.CalledProcessError: If the ffprobe command fails
        ValueError: If the creation_time metadata is missing or improperly formatted

    Note:
        Requires ffprobe (part of ffmpeg) to be installed and available in the system path
    """
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
    # print(f"{video_path}: {dt}")
    # print(dt.timestamp())
    return dt.timestamp()


def enhance_frame(image: np.ndarray) -> np.ndarray:
    """
    Apply advanced image processing techniques to make 7-segment display text more readable by LLMs,
    combining multiple approaches from various test files for optimal results.

    Args:
        image: The input image array

    Returns:
        Enhanced image array
    """
    # Convert to grayscale if image is in color
    if len(image.shape) == 3:
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = image

    # 1. Noise Reduction with Gaussian Blur (from test.py and test2.py)
    blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # 2. Contrast Enhancement with CLAHE (from test.py)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(blurred)

    # 3. Adaptive Thresholding (from test2.py)
    thresh = cv2.adaptiveThreshold(
        clahe_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    # 4. Morphological Operations: Opening to remove noise (from test2.py)
    kernel_open = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # 5. Connected Component Analysis to identify main text regions (from test2.py)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        opening, connectivity=8
    )

    # Find the largest component (excluding background)
    if num_labels > 1:  # Make sure we have at least one component besides background
        sizes = stats[1:, -1]  # Area of each component, excluding background
        max_label = 1 + np.argmax(sizes)

        # Create a mask for the largest component
        mask = np.zeros_like(gray_frame, dtype=np.uint8)
        mask[labels == max_label] = 255

        # Invert the mask to keep text area
        inverted_mask = cv2.bitwise_not(mask)

        # Apply mask to keep only the text region
        text_region = cv2.bitwise_and(gray_frame, gray_frame, mask=inverted_mask)
    else:
        # If no significant components found, use original grayscale
        text_region = gray_frame

    # 6. Invert for white text on black background (from test3.py)
    inverted = cv2.bitwise_not(text_region)

    # 7. Increase text thickness with dilation (from test3.py)
    kernel_dilate = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(inverted, kernel_dilate, iterations=1)

    # 8. Apply gamma correction for better contrast (from test3.py)
    gamma = 0.7  # Adjust gamma value as needed
    dilated_normalized = dilated / 255.0
    gamma_corrected = np.power(dilated_normalized, gamma)
    final_image = np.uint8(gamma_corrected * 255)

    return final_image


def extract_frames_at_interval(
    video_path: str, output_folder: str, interval_seconds=60
):
    """
    Extract frames from a video at specified intervals and save them as images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        date = extract_date(video_path)
    except Exception as e:
        print(f"Error extracting date from {video_path}, skipping: {e}")
        return
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

            # Apply enhancement filters to make the display more readable
            enhanced_frame = enhance_frame(cropped_frame)

            # Save the enhanced frame
            frame_filename = os.path.join(
                output_folder, f"{date + (round(frame_number / fps))}.jpg"
            )
            cv2.imwrite(frame_filename, enhanced_frame)
            print(f"Saved f{round(frame_number / fps)} seconds: {frame_filename}")

    # Release the video capture object
    cap.release()
    print(
        f"Extracted {len(frames_to_extract)} frames at {interval_seconds}-second intervals"
    )


def process_all_frames() -> None:
    """
    Process all frames in the extracted_frames directory and save enhanced versions
    to the modified_frames directory.
    """
    source_dir = "extracted_frames"
    target_dir = "modified_frames"

    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Count total images for progress tracking
    total_images = sum(
        len([f for f in files if f.lower().endswith(".jpg")])
        for _, _, files in os.walk(source_dir)
    )
    processed_count = 0

    print(f"Found {total_images} images to process")

    # Traverse all directories and files in the source directory
    for root, dirs, files in os.walk(source_dir):
        # Create corresponding directory in target
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, relative_path)

        if relative_path != ".":  # Skip the root directory
            if not os.path.exists(target_path):
                os.makedirs(target_path)

        # Process each image file
        for file in files:
            if file.lower().endswith(".jpg"):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_path, file)

                # Skip if file already exists in target
                if os.path.exists(target_file):
                    print(f"Skipping {file}, already exists in target")
                    processed_count += 1
                    continue

                try:
                    # Read the image
                    img = cv2.imread(source_file)
                    if img is None:
                        print(f"Failed to read {source_file}, skipping")
                        continue

                    # Enhance the image
                    enhanced_img = enhance_frame(img)

                    # Save the enhanced image
                    cv2.imwrite(target_file, enhanced_img)

                    processed_count += 1
                    if processed_count % 10 == 0 or processed_count == total_images:
                        print(f"Processed {processed_count}/{total_images} images")

                except Exception as e:
                    print(f"Error processing {source_file}: {e}")

    print(f"Complete! Enhanced {processed_count} images.")


if __name__ == "__main__":
    process_all_frames()
