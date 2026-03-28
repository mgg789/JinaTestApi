from __future__ import annotations

import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger("uvicorn.error")

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from emb import JinaEmbedder, JinaEmbeddingError, JinaTask

_API_USER = os.environ["API_USER"]
_API_PASSWORD = os.environ["API_PASSWORD"]

_security = HTTPBasic()

embedder: JinaEmbedder | None = None
_semaphore: asyncio.Semaphore | None = None


MODEL_PATH = Path("./models/jina-embeddings-v3")
MODEL_REPO = "jinaai/jina-embeddings-v3"


def _ensure_model() -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    if not (MODEL_PATH.is_dir() and any(MODEL_PATH.iterdir())):
        logger.info("Model not found locally, downloading from HuggingFace...")
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=str(MODEL_PATH),
        )
        logger.info("Model downloaded successfully.")

    # Pre-warm the transformers modules cache for custom code (trust_remote_code).
    # jina-embeddings-v3 pulls jinaai/xlm-roberta-flash-implementation at load time.
    # This must happen before JinaEmbedder sets HF_HUB_OFFLINE=1.
    logger.info("Pre-warming transformers module cache...")
    AutoConfig.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    logger.info("Module cache ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, _semaphore
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _ensure_model)
    embedder = JinaEmbedder()
    _semaphore = asyncio.Semaphore(1)  # one inference at a time to avoid OOM
    yield


app = FastAPI(title="Jina Embeddings API", lifespan=lifespan)


def _require_auth(credentials: HTTPBasicCredentials = Depends(_security)) -> None:
    ok = secrets.compare_digest(credentials.username, _API_USER) and \
         secrets.compare_digest(credentials.password, _API_PASSWORD)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


class EmbedRequest(BaseModel):
    text: Union[str, List[str]]
    task: Optional[JinaTask] = None
    batch_size: int = 16


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    shape: List[int]


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": embedder is not None}


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest, _: None = Depends(_require_auth)):
    if embedder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    loop = asyncio.get_event_loop()

    async with _semaphore:
        try:
            result = await loop.run_in_executor(
                None,
                lambda: embedder.embed(
                    request.text,
                    task=request.task,
                    batch_size=request.batch_size,
                ),
            )
        except JinaEmbeddingError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return EmbedResponse(
        embeddings=result.tolist(),
        shape=list(result.shape),
    )
