import os
import base64
from io import BytesIO
from PIL import Image
import requests
import uuid
from logger_module import logger
from prompts import IMAGE_STYLE_PROMPT

class RunwareImageAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.runware.ai/v1/image/inference"
        self.theme = "anime"  # Changed to cartoonish theme
        self.base_seed = 42
        self.max_prompt_length = 3000  # API max length
        self.min_prompt_length = 3     # API min length

    def generate_image(self, prompt: str, chunk_id: str, style: str = IMAGE_STYLE_PROMPT) -> str:
        # Base style text that will always be included
        base_style = f"{self.theme} style, bold outlines, vibrant colors, exaggerated features, playful and whimsical"
        
        # Calculate available length for the custom prompt
        base_style_length = len(base_style)
        available_length = self.max_prompt_length - base_style_length - 2  # -2 for ", " separator
        
        # Truncate the input prompt if necessary
        if len(prompt) > available_length:
            prompt = prompt[:available_length].rsplit(" ", 1)[0]  # Truncate at last full word
            logger.warning(f"Prompt truncated to {len(prompt)} characters to fit API limit of {self.max_prompt_length}.")
        
        # Ensure minimum length
        if len(prompt) < self.min_prompt_length:
            prompt = prompt.ljust(self.min_prompt_length, " ")  # Pad with spaces if too short
            logger.warning(f"Prompt padded to {self.min_prompt_length} characters to meet API minimum.")

        # Construct the final themed prompt
        themed_prompt = f"{base_style}, {prompt}"
        logger.debug(f"Generated prompt length: {len(themed_prompt)} characters")

        negative_prompt = "realistic, photorealistic, dark, dystopian, blurry, low quality"
        task_uuid = str(uuid.uuid4())
        payload = [
            {
                "taskType": "imageInference",
                "taskUUID": task_uuid,
                "model": "runware:100@1",  
                "positivePrompt": themed_prompt,
                "negativePrompt": negative_prompt,  
                "steps": 18,
                "width": 512,
                "height": 512,
                "numberResults": 1,
                "outputType": "base64Data",
                "seed": self.base_seed,
            }
        ]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            try:
                if isinstance(result, list) and result:
                    image_data = result[0]["imageBase64Data"]
                elif "data" in result and result["data"]:
                    image_data = result["data"][0]["imageBase64Data"]
                elif "imageBase64Data" in result:
                    image_data = result["imageBase64Data"]
                else:
                    raise KeyError("Could not find 'imageBase64Data' in response")
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
                os.makedirs("./image_out", exist_ok=True)
                image_path = f"./image_out/generated_image_{chunk_id}.png"
                image.save(image_path)
                return image_path
            except KeyError as e:
                logger.error(f"Error extracting image data: {str(e)}. Response: {result}")
                return ""
        else:
            logger.error(f"Runware API error: {response.status_code} - {response.text}")
            return ""