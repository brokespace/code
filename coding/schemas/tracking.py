from typing import List
from pydantic import BaseModel, Field


class TrackingInfo(BaseModel):
    logic: dict
    block: int  # deprecated
    hotkey: str
    uid: int
    raw_score: float = 0.0
    score: float = 0.0
    avg_cost: float = 0.0
    score_timestamps: List[int] = Field(
        default_factory=list
    )  # timestamp is the block number
