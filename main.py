import json
import time
import uuid
from pathlib import Path
from threading import Lock, Thread
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "Qwen/Qwen3-1.7B"
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="OpenAI-compatible Qwen API", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_tokenizer = None
_model = None
_model_lock = Lock()


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    stream: bool = False


def get_model_and_tokenizer():
    global _tokenizer, _model
    with _model_lock:
        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if _model is None:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.float32
            )
            _model.eval()
    return _model, _tokenizer


def build_prompt_messages(messages: list[ChatMessage]):
    return [{"role": m.role, "content": m.content} for m in messages]


def create_completion_response(
    model_name: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
):
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.get("/v1/models")
def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": now,
                "owned_by": "local",
            }
        ],
    }


@app.get("/")
def frontend():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if request.model != MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{request.model}'. Supported model: {MODEL_ID}",
        )

    model, tokenizer = get_model_and_tokenizer()
    chat_messages = build_prompt_messages(request.messages)

    prompt = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    if not request.stream:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=request.temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output[0][inputs["input_ids"].shape[-1] :]
        assistant_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completion_tokens = int(output[0].shape[-1] - inputs["input_ids"][0].shape[-1])
        return create_completion_response(
            model_name=request.model,
            content=assistant_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def event_stream():
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": 200,
            "do_sample": True,
            "temperature": request.temperature,
            "top_p": 0.9,
            "pad_token_id": tokenizer.eos_token_id,
        }

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"

        first_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

        for text in streamer:
            if not text:
                continue
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
