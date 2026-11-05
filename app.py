"""Minimal FastAPI app: uvicorn app:app --reload, then POST /ask {"question": "..."}."""
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from assistant.agent import TyreAssistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = FastAPI(title="Tyre care & fitment assistant (demo)")
bot = TyreAssistant()


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: AskIn):
    result = bot.ask(body.question)
    if result["route"] == "error":
        raise HTTPException(status_code=500, detail=result["answer"])
    return result
