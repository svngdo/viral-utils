from pydantic import BaseModel, Field


class FetchSelectedUserVideosRequest(BaseModel):
    user_ids: list[int] = Field(min_length=1)
