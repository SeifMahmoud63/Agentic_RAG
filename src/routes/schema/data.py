from pydantic import BaseModel
from typing import Optional,List

class ProcessRequest(BaseModel):
    chunk_size: Optional[int] = 100
    chunk_overlap: Optional[int] = 20
    do_reset: Optional[int] = 0
    file_ids: Optional[List[str]] = None 