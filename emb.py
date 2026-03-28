from __future__ import annotations

import os
import threading
from typing import Iterable, List, Literal, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


JinaTask = Literal[
    "retrieval.query",
    "retrieval.passage",
    "text-matching",
    "classification",
    "separation",
]

TextInput = Union[str, Sequence[str]]


class JinaEmbeddingError(Exception):
    pass


class InvalidTextInputError(JinaEmbeddingError):
    pass


class LocalModelNotFoundError(JinaEmbeddingError):
    pass


class JinaEmbedder:
    _TASK_TO_SUBFOLDER = {
        "retrieval.query": "retrieval_query",
        "retrieval.passage": "retrieval_passage",
        "text-matching": "text_matching",
        "classification": "classification",
        "separation": "separation",
    }

    def __init__(
        self,
        model_path: str = "./models/jina-embeddings-v3",
        device: Optional[str] = None,
        max_length: int = 2048,
        default_task: JinaTask = "text-matching",
        use_fp16: bool = True,
    ) -> None:
        if not os.path.isdir(model_path):
            raise LocalModelNotFoundError(
                f"Local model path does not exist: {model_path}"
            )

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        self.model_path = model_path
        self.max_length = max_length
        self.default_task = default_task
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.model.to(self.device)
        self.model.eval()

        if self.device.startswith("cuda") and use_fp16:
            self.model = self.model.half()

        self._loaded_adapters: set[str] = set()
        self._adapter_lock = threading.Lock()

    def embed(
        self,
        text_input: TextInput,
        task: Optional[JinaTask] = None,
        batch_size: int = 16,
        return_numpy: bool = True,
    ):
        texts = self._normalize_input(text_input)
        task_name = task or self.default_task

        if task_name not in self._TASK_TO_SUBFOLDER:
            raise ValueError(f"Unsupported task: {task_name}")

        self._set_task_adapter(task_name)

        outputs: List[torch.Tensor] = []

        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]

                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                model_output = self.model(**encoded)
                pooled = self._mean_pooling(model_output[0], encoded["attention_mask"])
                normalized = F.normalize(pooled, p=2, dim=1)
                outputs.append(normalized.detach().cpu())

        embeddings = torch.cat(outputs, dim=0)
        return embeddings.numpy() if return_numpy else embeddings

    def __call__(
        self,
        text_input: TextInput,
        task: Optional[JinaTask] = None,
        batch_size: int = 16,
        return_numpy: bool = True,
    ):
        return self.embed(
            text_input=text_input,
            task=task,
            batch_size=batch_size,
            return_numpy=return_numpy,
        )

    def _set_task_adapter(self, task_name: JinaTask) -> None:
        adapter_name = self._TASK_TO_SUBFOLDER[task_name]

        with self._adapter_lock:
            if adapter_name not in self._loaded_adapters:
                self.model.load_adapter(
                    self.model_path,
                    adapter_name=adapter_name,
                    adapter_kwargs={"subfolder": adapter_name},
                )
                self._loaded_adapters.add(adapter_name)

            self.model.set_adapter(adapter_name)

    @staticmethod
    def _mean_pooling(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @staticmethod
    def _normalize_input(text_input: TextInput) -> List[str]:
        if isinstance(text_input, str):
            text = text_input.strip()
            if not text:
                raise InvalidTextInputError("Input string is empty")
            return [text]

        if not isinstance(text_input, Iterable):
            raise InvalidTextInputError(
                "Input must be a string or a sequence of strings"
            )

        texts: List[str] = []
        for i, item in enumerate(text_input):
            if not isinstance(item, str):
                raise InvalidTextInputError(
                    f"Item at index {i} is not a string: {type(item).__name__}"
                )
            item = item.strip()
            if not item:
                raise InvalidTextInputError(
                    f"Item at index {i} is an empty string"
                )
            texts.append(item)

        if not texts:
            raise InvalidTextInputError("Input text list is empty")

        return texts