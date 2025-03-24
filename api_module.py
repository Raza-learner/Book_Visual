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
    IMAGE_STYLE_PROMPT
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

class RunwareImageAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.runware.ai/v1/image/inference"
        self.theme = "anime"  # Changed to cartoonish theme
        self.base_seed = 42
    def generate_image(self, prompt: str, chunk_id: str,    style: str = IMAGE_STYLE_PROMPT) -> str:
        task_uuid = str(uuid.uuid4())
        task_uuid = str(uuid.uuid4())
        themed_prompt = f"{self.theme} style, {prompt}, bold outlines, vibrant colors, exaggerated features, playful and whimsical"
        negative_prompt = "realistic, photorealistic, dark, dystopian, blurry, low quality"  # Exclude non-cartoonish elements
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

## HeadersSchema
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
    top_p: float = 0.8
    frequency_penalty: float = 1.0
    presence_penalty: float = 1.5

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

##Requests
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
    temperature: float = 0.4
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

    def get(self, messages: List[MessageSchema]) -> Tuple[int, str]:
        payload = SummaryPayloadSchema(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=self.stream,
        ).model_dump(by_alias=True)

        headers = HeadersSchema.create(api_key=self.api_key).model_dump(by_alias=True)
        response = requests.post(url=self.url, headers=headers, json=payload)
        code = response.status_code
        if code == 200:
            response_data = response.json()
            assistant_message = response_data["choices"][0]["message"]["content"]
            return code, assistant_message
        else:
            try:
                if response.json()["error"]["code"] == "json_validate_failed":
                    return 422, response.json()["error"]["failed_generation"]
            except:
                logger.warning(f"Error: {response.json()}")
                return code, "ERROR_API_CALL"

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
    content: List[Tuple[str, str]]
    summary: Summary
    image_api: Optional[RunwareImageAPI] = None
    summary_pool: List[SummaryOutputSchema] = Field(default_factory=list)
    chunked_content: List[Tuple[str, str, str]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def initialize(self) -> Optional["SummaryLoop"]:
        hf_api = os.environ.get("HF_API")
        if not hf_api:
            logger.error("[SummaryLoop] HF_API Not defined ")
            return None

        self.summary_pool = [
            SummaryOutputSchema(
                summary="This is The first chapeter There is No context",
                places={},
                characters={},
                id="",
                image_url=""
            ),
        ]
        tokenizer = Tokenizer(api_key=hf_api)
        logger.trace("tokenizer set")
        chunker = Chunker(max_len=self.summary.max_tokens, tokenizer=tokenizer)
        logger.trace("Chunker set")
        self.chunked_content = chunker.chunk(content=self.content)
        logger.trace("Chapters Chunked")

        return self

    def run(self) -> None:
        for idx, (id, title, content) in enumerate(self.chunked_content):
            past_context = self.summary_pool[idx]
            message = self.summary.get_messages(
                content=content,
                previous_summary=past_context.summary,
                characters=past_context.characters,
                places=past_context.places,
            )
            status_code, response = self.summary.get(messages=message)

            if status_code == 200:
                validated_response = self.summary.validate_json(
                    response, SummaryResponseSchema
                )

                if validated_response:
                    image_url = ""
                    if self.image_api:
                        # Use LLM output directly for the image prompt
                        summary = validated_response.summary
                        characters_str = ", ".join([f"{k} ({v})" for k, v in validated_response.characters.items()])
                        places_str = ", ".join([f"{k} ({v})" for k, v in validated_response.places.items()])
                        image_prompt = (
                            f"{summary} Featuring characters: {characters_str}. "
                            f"Set in places: {places_str}. "
                            
                        )
                        image_url = self.image_api.generate_image(image_prompt, chunk_id=id)
                    
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
            elif status_code == 422:
                output = self.handle_validation_error(response)
            elif status_code == 400:
                self.summary_pool.append(past_context)
                logger.warning(f"{status_code=} error getting{id=}")

    def handle_validation_error(self, input_text):
        message = self.summary.validation_messages(input_text)
        for idx in range(MAX_VALIDATION_ERROR_TRY):
            status_code, response = self.summary.get(messages=message)
            if status_code == 200:
                validated_response = self.summary.validate_json(
                    response, SummaryResponseSchema
                )
                if validated_response:
                    logger.info("Validation error resolved")
                    return validated_response
            elif status_code == 422:
                message = self.summary.validation_messages(response)
            logger.warning(f"Validation Unresolved on try {idx + 1}")
        logger.error("COULDNT VALIDATE THE CHUNK, SKIPPING...")
        return None

    @property
    def get_summary_pool(self):
        return self.summary_pool

async def test() -> None:
    from reader import ebook

    api = os.environ.get("GROQ_API")
    runware_api_key = os.environ.get("RUNWARE_API")
    if not api:
        raise Exception("API NOT SET IN .env, GROQ_API=None")
    if not runware_api_key:
        raise Exception("RUNWARE_API NOT SET IN .env")

    image_api = RunwareImageAPI(api_key=runware_api_key)

    book = ebook("./HP.epub")
    chapter_content = book.get_chapters()[1:6]
    sum = Summary(api_key=api)
    from audio_module import loop_for_speech

    looper = SummaryLoop(content=chapter_content, summary=sum, image_api=image_api).initialize()
    if not looper:
        return None
    looper.run()
    
    summary_texts = []
    for i in looper.get_summary_pool:
        if not i.summary:
            continue
        text = f"Chapter {i.id}. Summary: {i.summary}"
        summary_texts.append((i.id, text))

    if summary_texts:
        audio_outputs = loop_for_speech(summary_texts)
        os.makedirs("./audio_out", exist_ok=True)
        for idx, (chapter_id, audio_content) in enumerate(audio_outputs):
            if audio_content:
                audio_file = f"./audio_out/summary_chapter_{chapter_id}.mp3"
                with open(audio_file, "wb") as file:
                    file.write(audio_content)
                logger.info(f"Audio saved for Chapter {chapter_id} at {audio_file}")
            else:
                logger.warning(f"Audio generation failed for Chapter {chapter_id}")

    for i in looper.get_summary_pool:
        if not i.places or not i.characters:
            continue
        print("-" * 50)
        print(f"Chapter:{i.id}")
        print("Summary")
        print(f"\t - {i.summary}")
        print(f"Image Path: {i.image_url}")
        print()
        print("characters")
        for k, v in i.characters.items():
            print(f"\t - {k} : {v}")
        print()
        print("places")
        for k, v in i.places.items():
            print(f"\t - {k} : {v}")
        print("\n\n")

if __name__ == "__main__":
    asyncio.run(test())
