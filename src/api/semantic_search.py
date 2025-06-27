from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from core.semantic.code_search import search as csn_search

router = APIRouter(prefix="/api/v1/semantic", tags=["semantic"])

class CodeSearchRequest(BaseModel):
    snippet: str = Field(..., min_length=5, max_length=4096)
    k: int = Field(5, ge=1, le=20)

class CodeSearchHit(BaseModel):
    similarity: float
    metadata: Dict

class CodeSearchResponse(BaseModel):
    hits: List[CodeSearchHit]

@router.post("/search", response_model=CodeSearchResponse)
async def semantic_code_search(req: CodeSearchRequest):
    try:
        hits = csn_search(req.snippet, req.k)
        return {"hits": [{"similarity": s, "metadata": m} for s, m in hits]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
