from typing import Protocol

from google import genai
from openai import OpenAI
from src.config import settings
from src.llm.exceptions import EmptyResponseError, LLMError, normalize_error


class LLMProvider(Protocol):
    model_id: str

    async def generate(
        self,
        instruction: str,
        prompt: str,
        temperature: float | None = None,
    ) -> str: ...


class GeminiProvider:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=settings.gemini_api_key)
        return self._client

    async def generate(
        self,
        instruction: str,
        prompt: str,
        temperature: float | None = 0.3,
    ) -> str:
        try:
            response = self._get_client().models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=instruction,
                    temperature=temperature,
                ),
            )
            if response.text is None:
                raise EmptyResponseError("Empty response")
            return response.text.strip()
        except LLMError:
            raise
        except Exception as e:
            raise normalize_error(e)


class OpenAICompatProvider:
    def __init__(self, model_id: str, api_key: str, base_url: str):
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    async def generate(
        self,
        instruction: str,
        prompt: str,
        temperature: float | None = 0.3,
    ) -> str:
        try:
            response = self._get_client().chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            content = response.choices[0].message.content
            if content is None:
                raise EmptyResponseError("Empty response")
            return content.strip()
        except LLMError:
            raise
        except Exception as e:
            raise normalize_error(e)
