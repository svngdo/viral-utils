from src.llm.provider import LLMProvider
from src.logging import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(self, providers: list[LLMProvider]):
        self._providers = providers

    async def generate(self, instruction: str, prompt: str) -> str:
        errors = []
        for provider in self._providers:
            try:
                result = await provider.generate(instruction=instruction, prompt=prompt)
                logger.info(f"LLM response via {provider.model_id}")
                return result
            except Exception as e:
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                logger.debug(
                    f"Skipping {provider.model_id} ({code or type(e).__name__})"
                )
                errors.append(f"{provider.model_id}: {code or e}")
        raise RuntimeError("All models failed:\n" + "\n".join(errors))
