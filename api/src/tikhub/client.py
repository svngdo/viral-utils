import asyncio
import logging
from typing import Any

import httpx
from pydantic import ValidationError

from src.rate_limit.bucket import TokenBucket
from src.tikhub.constants import (
    DEFAULT_HEADERS,
    DEFAULT_USER_POST_VIDEO_COUNT,
    TIKHUB_BASE_URL,
)
from src.tikhub.exceptions import (
    TikHubRequestError,
    TikHubStatusError,
    TikHubValidationError,
)
from src.tikhub.schemas import UserPostVideosResponse

logger = logging.getLogger(__name__)


class TikHubClient:
    def __init__(self):
        self._limiter = TokenBucket(capacity=5, refill_rate=1.0)

    async def fetch_multi_video(self, video_ids: list[str]) -> Any:
        url = TIKHUB_BASE_URL + "/fetch_multi_video"
        headers = DEFAULT_HEADERS

        await self._limiter.acquire()
        async with httpx.AsyncClient() as client:
            response = await client.post(url=url, headers=headers, json=video_ids)
            return response.json()

    async def fetch_user_post_videos(
        self,
        sec_uid: str,
        max_cursor: int = 0,
        count: int = DEFAULT_USER_POST_VIDEO_COUNT,
        sort_type: int = 0,
    ) -> UserPostVideosResponse:
        url = TIKHUB_BASE_URL + "/fetch_user_post_videos"
        headers = DEFAULT_HEADERS
        params = {
            "sec_user_id": sec_uid,
            "max_cursor": max_cursor,
            "count": count,
            "sort_type": sort_type,
        }

        try:
            await self._limiter.acquire()
            async with httpx.AsyncClient() as client:
                res = await client.get(url=url, headers=headers, params=params)
                res.raise_for_status()
                return UserPostVideosResponse.model_validate(res.json())
        except httpx.HTTPStatusError as e:
            detail = e.response.text
            logger.warning(
                "TikHub status error - status=%s - detail=%s",
                e.response.status_code,
                detail,
            )
            raise TikHubStatusError(
                message=f"TikHub returned {e.response.status_code}: {detail}",
                upstream_status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            logger.exception(e)
            raise TikHubRequestError() from e
        except ValidationError as e:
            logger.exception(e)
            raise TikHubValidationError() from e
        except Exception:
            raise

    async def fetch_multi_user_post_videos(
        self,
        sec_user_ids: list[str],
        max_cursor: int,
        count: int,
        sort_type: int = 0,
    ) -> list[UserPostVideosResponse]:
        tasks = [
            self.fetch_user_post_videos(
                sec_uid=id,
                max_cursor=max_cursor,
                count=count,
                sort_type=sort_type,
            )
            for id in sec_user_ids
        ]
        return await asyncio.gather(*tasks)
