"""
FastAPI Backend — Smart Paraphrasing Tool

Endpoints:
  POST /paraphrase  — paraphrase input text
  GET  /health      — health check
  GET  /            — serve web UI
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import time
import os

from pipeline import paraphrase_paragraph

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GenAI Smart Paraphrasing Tool",
    description="T5-based paragraph paraphrasing with multi-metric ranking",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files and templates
os.makedirs("static",    exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ─── Request / Response Models ────────────────────────────────────────────────
class ParaphraseRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=2000,
                      example="Artificial intelligence is transforming industries worldwide.")
    num_beams: int = Field(default=5, ge=2, le=10)
    num_candidates: int = Field(default=5, ge=2, le=10)
    level: str = Field(default="Standard")

class SentenceDetail(BaseModel):
    original: str
    paraphrased: str
    semantic_similarity: float
    fluency_score: float
    readability_score: float
    diversity_score: float
    total_score: float

class ParaphraseResponse(BaseModel):
    original_text: str
    paraphrased_text: str
    sentence_count: int
    sentence_details: list[SentenceDetail]
    processing_time_ms: float

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    return {"status": "ok", "model": "t5-based-paraphraser"}

@app.post("/paraphrase", response_model=ParaphraseResponse)
async def paraphrase(req: ParaphraseRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    start = time.time()
    try:
        result = paraphrase_paragraph(
            req.text,
            num_beams=req.num_beams,
            num_return_sequences=req.num_candidates,
            level=req.level,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    elapsed_ms = (time.time() - start) * 1000

    sentence_details = []
    for detail in result["sentence_details"]:
        s = detail["scores"]
        sentence_details.append(SentenceDetail(
            original=detail["original"],
            paraphrased=detail["paraphrased"],
            semantic_similarity=s["semantic_similarity"],
            fluency_score=s["fluency_score"],
            readability_score=s["readability_score"],
            diversity_score=s["diversity_score"],
            total_score=s["total_score"],
        ))

    return ParaphraseResponse(
        original_text=result["original_text"],
        paraphrased_text=result["paraphrased_text"],
        sentence_count=result["sentence_count"],
        sentence_details=sentence_details,
        processing_time_ms=round(elapsed_ms, 1),
    )

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
