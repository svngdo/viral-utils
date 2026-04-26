from src.llm.client import LLMClient
from src.llm.registry import build_free_providers
from src.video.config import VideoConfig, video_config


def get_llm_client() -> LLMClient:
    return LLMClient(providers=build_free_providers())


def get_video_config() -> VideoConfig:
    return video_config
