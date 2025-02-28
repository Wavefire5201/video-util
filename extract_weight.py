from pydantic import BaseModel
from dotenv import load_dotenv
import os
from google import genai
import PIL.Image
from ollama import chat, ChatResponse

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class Weight(BaseModel):
    weight: float


# so far minicpm-v has about 98% accuracy
def extract_weight_ollama(image_path) -> float:
    response: ChatResponse = chat(
        messages=[
            {
                "role": "user",
                "content": "Extract the weight from this image in a float format. Make sure it is exactly correct. Do not leave out any digits. If it's an seven segmented display, ignore any possible translucent segments. Do not confused any place values.",
                "images": [image_path],
            }
        ],
        options={"temperature": 0.2},
        model="minicpm-v",
        format=Weight.model_json_schema(),
    )
    return Weight.model_validate_json(response.message.content).weight


# gemini 2.0 flash has 100% accuracy so far
def extract_weight_gemini(image_path) -> float:
    image = PIL.Image.open(image_path)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=["Extract the weight from this image in a float format.", image],
        config={
            "response_mime_type": "application/json",
            "response_schema": Weight,
        },
    )

    return response.parsed.weight


# for filename in sorted(os.listdir("extracted_frames")):
#     if filename.endswith(".jpg"):
#         weight = extract_weight_gemini("extracted_frames/" + filename)
#         print(f"{filename}: {weight}")
