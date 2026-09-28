import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient


app = FastAPI(
    title="Qwen QLoRA Inference API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_ID = "shivamkumar0502/qwen2.5-1.5b-dolly-qlora"


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200


@app.get("/")
def root():
    return {
        "status": "running",
        "model": BASE_MODEL,
        "adapter": ADAPTER_ID
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/generate")
def generate(request: GenerateRequest):

    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="HF_TOKEN is not configured."
        )

    if not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty."
        )

    try:
        client = InferenceClient(token=HF_TOKEN)

        result = client.text_generation(
            prompt=request.prompt,
            model=BASE_MODEL,
            adapter_id=ADAPTER_ID,
            max_new_tokens=request.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,
        )

        return {
            "response": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
