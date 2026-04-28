import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from query_engine import ask

app = FastAPI(title="Softball Rules Assistant")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/ask", response_model=Answer)
async def ask_question(q: Question):
    answer = ask(q.question)
    return Answer(answer=answer)
