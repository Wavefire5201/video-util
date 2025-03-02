from pydantic import BaseModel
from dotenv import load_dotenv
import os
from google import genai
from PIL import Image
from ollama import chat, ChatResponse

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class Weight(BaseModel):
    """
    A Pydantic model representing a weight value.
    """

    weight: float


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


def extract_weight_gemini(image_path: str) -> float:
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
        image = Image.open(image_path)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Extract the weight from this image in a float format.", image],
            config={
                "response_mime_type": "application/json",
                "response_schema": Weight,
            },
        )
        return response.parsed.weight
    except Exception as e:
        raise Exception(f"Error in extract_weight_gemini: {str(e)}")
