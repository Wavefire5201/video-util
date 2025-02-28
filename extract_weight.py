import cv2
import pytesseract
import re
from pydantic import BaseModel
import os
from google import genai
from dotenv import load_dotenv
import PIL.Image
import natsort

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# change based on session, might be different for each session
x = 577
y = 528
width = 815
height = 365


class WeightExtractionResult(BaseModel):
    weight: float


def crop_image(image_path) -> PIL.Image.Image:
    image = PIL.Image.open(image_path)
    cropped_image = image.crop((x, y, x + width, y + height))
    return cropped_image


def extract_weight(image_path) -> float:
    image = crop_image(image_path)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=["Extract the weight from this image in a float format.", image],
        config={
            "response_mime_type": "application/json",
            "response_schema": WeightExtractionResult,
        },
    )

    return response.parsed.weight


def extract_weight_cv(image_path):
    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None

    # Crop the image to the region of interest
    img = img[y : y + height, x : x + width]
    print(pytesseract.image_to_string(img))
    print("it worked here")
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to improve OCR accuracy
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Try different preprocessing techniques
    # Noise removal and dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Try multiple configurations for OCR
    configs = [
        "--psm 7 -c tessedit_char_whitelist=-.0123456789g",
        "--psm 8 -c tessedit_char_whitelist=-.0123456789g",
        "--psm 6 -c tessedit_char_whitelist=-.0123456789g",
        "--psm 10 -c tessedit_char_whitelist=-.0123456789g",
    ]

    for config in configs:
        # Try original grayscale
        text_gray = pytesseract.image_to_string(gray, config=config)
        weight_gray = re.search(r"(-?\d+\.\d+)", text_gray)
        if weight_gray:
            return float(weight_gray.group(1))

        # Try threshold
        text_thresh = pytesseract.image_to_string(thresh, config=config)
        weight_thresh = re.search(r"(-?\d+\.\d+)", text_thresh)
        if weight_thresh:
            return float(weight_thresh.group(1))

        # Try opening
        text_opening = pytesseract.image_to_string(opening, config=config)
        weight_opening = re.search(r"(-?\d+\.\d+)", text_opening)
        if weight_opening:
            return float(weight_opening.group(1))

    # If we couldn't find a weight with specific configurations, try a more general approach
    text = pytesseract.image_to_string(img)
    print("Full OCR output:", text)

    # Try different regex patterns
    patterns = [r"(-?\d+\.\d+)", r"(\d+\.\d+)", r"(\d+)"]

    for pattern in patterns:
        weight_match = re.search(pattern, text)
        if weight_match:
            return float(weight_match.group(1))

    return None


# for each image in the folder, extract weight
# open folder
# for filename in os.listdir("extracted_frames"):
#     if filename.endswith(".jpg"):
#         weight = extract_weight("extracted_frames/" + filename)
#         print(f"{filename}: {weight}")

extract_weight_cv("extracted_frames/frame_1_0-00-00.jpg")
