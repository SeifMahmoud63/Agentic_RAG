from pydantic import BaseModel
from typing import Optional,List

class ProcessRequest(BaseModel):
    chunk_size: Optional[int] 
    chunk_overlap: Optional[int] 
    do_reset: Optional[int] = 0
