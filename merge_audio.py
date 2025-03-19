from typing import Dict, List, Optional, Type, Tuple, Union
from pydantic import BaseModel, Field, ValidationError
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from logger_module import logger
from utils import Chunker, Tokenizer
import requests
import asyncio
from prompts import (
    MAX_VALIDATION_ERROR_TRY,
    SUMMARY_ROLE,
    SUMMARY_VALIDATION_RESOLVE_ROLE,
)
from json.decoder import JSONDecodeError
import os
from dotenv import load_dotenv
import time
import base64
from io import BytesIO
from PIL import Image
import uuid

load_dotenv()
summary_role = ""

# Updated RunwareImageAPI with chunk ID in image name
class RunwareImageAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.runware.ai/v1/image/inference"

    def generate_image(self, prompt: str, chunk_id: str) -> str:
        task_uuid = str(uuid.uuid4())  # Still needed for the API payload
        payload = [
            {
                "taskType": "imageInference",
                "taskUUID": task_uuid,
                "model": "runware:100@1",
                "positivePrompt": prompt,
                "steps": 4,
                "width": 1024,
                "height": 1024 ,
                "numberResults": 1,
                "outputType": "base64Data",
                
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
                
                # Decode and save the image with chunk_id in the name
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
                image_path = f"generated_image_{chunk_id}.png"  # Use chunk_id in filename
                image.save(image_path)
                return image_path
            except KeyError as e:
                logger.error(f"Error extracting image data: {str(e)}. Response: {result}")
                return ""
        else:
            logger.error(f"Runware API error: {response.status_code} - {response.text}")
            return ""

# HeadersSchema
class HeadersSchema(BaseModel):
    authorization: str = Field(..., alias="Authorization")
    content_type: str = Field(default="application/json", alias="Content-Type")

    @classmethod
    def create(cls, api_key: str) -> "HeadersSchema":
        return cls(Authorization=f"Bearer {api_key}")

    class Config:
        populate_by_name = True

class MessageSchema(BaseModel):
    role: str
    content: str

class SummaryPayloadSchema(BaseModel):
    model: str
    messages: List[MessageSchema]
    temperature: float
    stream: bool
    max_completion_tokens: int = 2048
    response_format: Dict[str, str] = {"type": "json_object"}

class SummaryResponseSchema(BaseModel):
    summary: str
    characters: Dict[str, str]
    places: Dict[str, str]

class SummaryOutputSchema(SummaryResponseSchema):
    id: str
    image_url: str = ""

class SummaryContentSchema(BaseModel):
    past_context: str
    current_chapter: str
    character_list: Dict[str, str]
    places_list: Dict[str, str]

class LLM_API(ABC):
    @abstractmethod
    def messages(
        self, content: str, character: Dict[str, str], places: Dict[str, str]
    ) -> List[MessageSchema]:
        pass

    @abstractmethod
    def get(self, message: List[MessageSchema]) -> str:
        pass

    @abstractmethod
    def validate_json(
        self, raw_data: str, schema: Type[BaseModel]
    ) -> Optional[Type[BaseModel]]:
        pass

@dataclass
class Summary:
    api_key: str
    url: str = "https://api.groq.com/openai/v1/chat/completions"
    role: str = (
        f"{SUMMARY_ROLE} follow given schema: {SummaryOutputSchema.model_json_schema()}"
    )
    validation_role: str = f"{SUMMARY_VALIDATION_RESOLVE_ROLE} Schema :{SummaryOutputSchema.model_json_schema()}"
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    stream: bool = False
    repetition_penalty: float = 1.5
    max_tokens: int = 6000

    def get_messages(
        self,
        content: str,
        previous_summary: str,
        characters: Dict[str, str],
        places: Dict[str, str],
    ) -> List[MessageSchema]:
        return [
            MessageSchema(role="system", content=self.role),
            MessageSchema(
                role="user",
                content=SummaryContentSchema(
                    past_context=previous_summary,
                    current_chapter=content,
                    character_list=characters,
                    places_list=places,
                ).model_dump_json(by_alias=True),
            ),
        ]

    def validation_messages(self, input_text: str) -> List[MessageSchema]:
        return [
            MessageSchema(role="system", content=self.validation_role),
            MessageSchema(
                role="user",
                content=input_text,
            ),
        ]

    def get(self, messages: List[MessageSchema]) -> Tuple[bool, str]:
        payload = SummaryPayloadSchema(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=self.stream,
        ).model_dump(by_alias=True)

        headers = HeadersSchema.create(api_key=self.api_key).model_dump(by_alias=True)
        response = requests.post(url=self.url, headers=headers, json=payload)
        if response.status_code == 200:
            response_data = response.json()
            assistant_message = response_data["choices"][0]["message"]["content"]
            return False, assistant_message
        else:
            logger.warning(f"Error: {response.json()}")
            return True, "ERROR_API_CALL"

    def validate_json(
        self, raw_data: str, schema: Type[SummaryResponseSchema]
    ) -> Union[SummaryResponseSchema, bool]:
        try:
            parsed_data = json.loads(raw_data)
            validated_data = schema.model_validate(parsed_data)
            return validated_data
        except (ValidationError, JSONDecodeError):
            logger.warning("ValidationError")
            return False

class SummaryLoop(BaseModel):
    content: List[Tuple[str, str]]  # (id, content)
    summary: Summary
    image_api: Optional[RunwareImageAPI] = None
    summary_pool: List[SummaryOutputSchema] = Field(default_factory=list)
    chunked_content: List[Tuple[str, str, str]] = Field(default_factory=list)  # (id, title, content)

    class Config:
        arbitrary_types_allowed = True

    def initialize(self) -> Optional["SummaryLoop"]:
        self.chunked_content = [
            (id, f"Chapter {i+1}", content)
            for i, (id, content) in enumerate(self.content)
        ]
        return self

    def run(self) -> None:
        for idx, (id, title, content) in enumerate(self.chunked_content):
            past_context = self.summary_pool[idx] if idx < len(self.summary_pool) else SummaryOutputSchema(
                id=id, summary="", characters={}, places={}, image_url=""
            )
            message = self.summary.get_messages(
                content=content,
                previous_summary=past_context.summary,
                characters=past_context.characters,
                places=past_context.places,
            )
            error, response = self.summary.get(messages=message)

            if not error:
                validated_response = self.summary.validate_json(
                    response, SummaryResponseSchema
                )

                if validated_response:
                    image_url = ""
                    if self.image_api:
                        image_prompt = f"Book chapter illustration for: {validated_response.summary}"
                        image_url = self.image_api.generate_image(image_prompt, chunk_id=id)  # Pass chunk ID
                    
                    logger.trace(f"Chunk_{id=} Done")
                    self.summary_pool.append(
                        SummaryOutputSchema(
                            **(validated_response.model_dump(by_alias=True)),
                            id=id,
                            image_url=image_url
                        )
                    )
                    continue
                else:
                    output = self.handle_validation_error(response)
                    if output:
                        past_context = output

            self.summary_pool.append(past_context)
            logger.warning(f"error getting {id=}")

    def handle_validation_error(self, input_text):
        message = self.summary.validation_messages(input_text)
        for _ in range(MAX_VALIDATION_ERROR_TRY):
            err, response = self.summary.get(messages=message)
            if not err:
                validated_response = self.summary.validate_json(
                    response, SummaryResponseSchema
                )
                if validated_response:
                    return validated_response
        logger.error("COULDNT VALIDATE THE CHUNK, SKIPPING...")
        return None

    @property
    def get_summary_pool(self):
        return self.summary_pool

async def test() -> None:
    from reader import ebook
    runware_api = RunwareImageAPI(api_key=os.environ.get("RUNWARE_API_KEY"))
    api = os.environ.get("GROQ_API")
    if not api:
        raise Exception("API NOT SET IN .env, HF_API=None")

    book = ebook("./HP.epub")
    chapter_content = book.get_chapters()
    print("Chapter content:", chapter_content)

    sum = Summary(
        api_key=api,
    )

    looper = SummaryLoop(content=chapter_content, summary=sum, image_api=runware_api).initialize()
    if not looper:
        return None
    looper.run()
    for i in looper.get_summary_pool:
        if not i.places or not i.characters:
            continue
        print("-" * 50)
        print(f"Chapter:{i.id}")
        print("Summary")
        print(f"\t - {i.summary}")
        print()
        print("characters")
        print(f"Image Path (Generated with civitai:7240@119057): {i.image_url}\n")
        for k, v in i.characters.items():
            print(f"\t - {k} : {v}")
        print()
        print("places")
        for k, v in i.places.items():
            print(f"\t - {k} : {v}")
        print("\n\n")

if __name__ == "__main__":
    asyncio.run(test())
