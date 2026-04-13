from pydantic import BaseModel,Field
from typing import Optional,List
from helpers import config

class ProcessRequest(BaseModel):
    chunk_size: Optional[int] = Field(
        default_factory=lambda: config.get_settings().chunk_size
    )
    chunk_overlap: Optional[int] = Field(
        default_factory=lambda: config.get_settings().chunk_overlap
    )
    file_ids: Optional[List[str]] = None