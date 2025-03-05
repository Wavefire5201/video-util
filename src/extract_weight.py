import base64
import os

import openai
from dotenv import load_dotenv
from google import genai
from ollama import ChatResponse, chat
from PIL import Image
from pydantic import BaseModel

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
or_client = openai.Client(
    base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
)


class Weight(BaseModel):
    """
    A Pydantic model representing a weight value (in grams).
    """

    weight: float


def encode_image(image_path):
    """Encodes an image to base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_weight_ollama(image_path: str) -> float:
    """
    Extracts weight from an image using the Ollama minicpm-v model.

    Args:
        image_path (str): Path to the image file containing the weight.

    Returns:
        float: The extracted weight value.

    Raises:
        Exception: If any error occurs during processing.
    """
    try:
        response: ChatResponse = chat(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract the weight from this image in a float format. "
                        "Make sure it is exactly correct. Do not leave out any digits. "
                        "If it's a seven-segment display, ignore any possible translucent segments. "
                        "Do not confuse any place values."
                    ),
                    "images": [image_path],
                }
            ],
            options={"temperature": 0.2},
            model="minicpm-v",
            format=Weight.model_json_schema(),
        )
        return Weight.model_validate_json(response.message.content).weight
    except Exception as e:
        raise Exception(f"Error in extract_weight_ollama: {str(e)}")


def extract_weight_gemini(image_path: str, previous_result: float = 0.0) -> float:
    """
    Extracts weight from an image using the Gemini 2.0 flash model.

    Args:
        image_path (str): Path to the image file containing the weight.

    Returns:
        float: The extracted weight value.

    Raises:
        Exception: If any error occurs during processing.
    """
    try:
        print(
            f"Extracting weight from {image_path} with previous result {previous_result}"
        )
        image = Image.open(image_path)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                f"What's the number in this image? Return a singular float. Properly distinguish the place value. The image is of a seven-segmented display of a scale from a video extracted frame by frame. The previous value extracted is {previous_result}. The resulting float should be VERY close to the previous value.",
                image,
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": Weight,
            },
        )
        return response.parsed.weight
    except Exception as e:
        raise Exception(f"Error in extract_weight_gemini: {str(e)}")


def extract_weight_openrouter(image_path: str, previous_result: float = 0.0) -> float:
    """
    Extracts weight from an image using OpenRouter.

    Args:
        image_path (str): Path to the image file containing the weight.

    Returns:
        float: The extracted weight value.

    Raises:
        Exception: If any error occurs during processing.
    """
    try:
        print(
            f"Extracting weight from {image_path} with previous result {previous_result}"
        )
        image = encode_image(image_path)
        response = or_client.beta.chat.completions.parse(
            model="google/gemini-2.0-flash-exp:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"What's the number in this image? Return a singular float. Properly distinguish the place value. The image is of a seven-segmented display of a scale from a video extracted frame by frame. The previous value extracted is {previous_result}. The resulting float should be VERY close to the previous value.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                        },
                    ],
                }
            ],
            response_format=Weight,
        )
        return response.choices[0].message.parsed.weight

    except Exception as e:
        raise Exception(f"Error in extract_weight_openrouter: {str(e)}")
