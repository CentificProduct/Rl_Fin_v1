"""
===========================================================================
LLM Provider Abstraction
===========================================================================

Unified interface for calling LLMs from multiple backends:

  OllamaProvider      -- local Ollama server (llama3, qwen2.5, mistral)
  HuggingFaceProvider -- transformers library (any HF model, 4-bit quantized)
  MockProvider        -- deterministic responses for testing without LLM

Usage:
    provider = get_provider("ollama", "qwen2.5")
    provider = get_provider("huggingface", "Qwen/Qwen2.5-3B-Instruct")
    provider = get_provider("mock")       # always works, no deps

    response = provider.chat([{"role": "user", "content": "..."}])
    score = provider.chat([...], json_mode=True)  # structured JSON output
    embeddings = provider.embed(["headline 1", "headline 2"])
===========================================================================
"""

import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Base interface that all LLM backends must implement."""

    model_name: str = "unknown"

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        """Send a chat-style prompt and return the response text."""
        ...

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a single prompt (no history) and return the response."""
        ...

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return embedding vectors for a list of texts. Optional."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support embeddings")

    def is_available(self) -> bool:
        """Check if this provider is ready to serve requests."""
        try:
            self.generate("Say OK")
            return True
        except Exception:
            return False

    def chat_json(self, messages: List[Dict[str, str]]) -> Dict:
        """Chat with JSON-mode and parse the result into a dict."""
        raw = self.chat(messages, json_mode=True)
        return _parse_json_response(raw)


# =====================================================================
# Ollama Backend
# =====================================================================

class OllamaProvider(LLMProvider):
    """
    Local LLM via Ollama (http://localhost:11434).

    Requires: ollama Python package OR a running Ollama server.
    Models: qwen2.5, llama3.2, mistral, etc.
    """

    def __init__(self, model_name: str = "qwen2.5", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama
            except ImportError:
                self._client = "requests"
        return self._client

    def chat(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        client = self._get_client()

        if client != "requests":
            kwargs = {"model": self.model_name, "messages": messages}
            if json_mode:
                kwargs["format"] = "json"
            response = client.chat(**kwargs)
            return response["message"]["content"]
        else:
            return self._chat_via_requests(messages, json_mode)

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        if client != "requests":
            response = client.generate(model=self.model_name, prompt=prompt)
            return response["response"]
        else:
            return self._generate_via_requests(prompt)

    def embed(self, texts: List[str]) -> np.ndarray:
        client = self._get_client()
        if client != "requests":
            response = client.embed(model=self.model_name, input=texts)
            return np.array(response["embeddings"])
        else:
            return self._embed_via_requests(texts)

    def is_available(self) -> bool:
        try:
            import requests as req
            resp = req.get(f"{self.base_url}/api/version", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def _chat_via_requests(self, messages, json_mode):
        import requests as req
        payload = {"model": self.model_name, "messages": messages, "stream": False}
        if json_mode:
            payload["format"] = "json"
        resp = req.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _generate_via_requests(self, prompt):
        import requests as req
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}
        resp = req.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"]

    def _embed_via_requests(self, texts):
        import requests as req
        payload = {"model": self.model_name, "input": texts}
        resp = req.post(f"{self.base_url}/api/embed", json=payload, timeout=60)
        resp.raise_for_status()
        return np.array(resp.json()["embeddings"])


# =====================================================================
# HuggingFace Transformers Backend
# =====================================================================

class HuggingFaceProvider(LLMProvider):
    """
    Direct model loading via HuggingFace transformers.

    Supports 4-bit quantization for running large models on consumer GPUs.
    Models: Qwen/Qwen2.5-3B-Instruct, mistralai/Mistral-7B-Instruct-v0.3, etc.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", quantize_4bit: bool = True):
        self.model_name = model_name
        self.quantize_4bit = quantize_4bit
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if self.quantize_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except ImportError:
                logger.warning("bitsandbytes not available, loading without quantization")

        logger.info(f"Loading model {self.model_name} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs
        )
        logger.info("Model loaded.")

    def chat(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        self._load_model()
        import torch

        if json_mode:
            messages = list(messages)
            last = messages[-1]
            messages[-1] = {
                **last,
                "content": last["content"] + "\n\nRespond ONLY with valid JSON.",
            }

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=512, do_sample=True,
                temperature=0.7, top_p=0.9,
            )
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def generate(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False


# =====================================================================
# Mock Provider (for testing / CI)
# =====================================================================

class MockProvider(LLMProvider):
    """
    Deterministic mock that returns canned responses.
    Works without any LLM -- useful for testing and benchmarks.
    """

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        self._call_count = 0

    def chat(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        self._call_count += 1
        content = messages[-1]["content"] if messages else ""

        if json_mode or "json" in content.lower():
            return self._mock_json_response(content)
        return self._mock_text_response(content)

    def generate(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])

    def embed(self, texts: List[str]) -> np.ndarray:
        np.random.seed(42)
        return np.random.randn(len(texts), 384).astype(np.float32)

    def is_available(self) -> bool:
        return True

    def _mock_json_response(self, content: str) -> str:
        c = content.lower()
        if "sentiment" in c or "score" in c or "headline" in c:
            return json.dumps({
                "sentiment": 0.15,
                "risk": 0.3,
                "confidence": 0.7,
            })
        if "action" in c or "trade" in c or "decision" in c:
            return json.dumps({
                "action": "hold",
                "confidence": 0.6,
                "reasoning": "Market conditions are neutral with moderate volatility.",
            })
        if "scenario" in c or "predict" in c or "forecast" in c:
            if "array" in c or "generate" in c and any(w in c for w in ["3", "5", "multiple"]):
                return json.dumps([
                    {"direction": "up", "magnitude": 0.008, "volatility_change": -0.01,
                     "probability": 0.45, "reasoning": "Momentum continuation with declining vol."},
                    {"direction": "flat", "magnitude": 0.001, "volatility_change": 0.0,
                     "probability": 0.30, "reasoning": "Consolidation near current levels."},
                    {"direction": "down", "magnitude": -0.005, "volatility_change": 0.02,
                     "probability": 0.25, "reasoning": "Profit-taking with rising volatility."},
                ])
            return json.dumps({
                "direction": "up",
                "magnitude": 0.005,
                "volatility_change": 0.0,
                "probability": 0.5,
                "reasoning": "Slight upward momentum with stable volatility.",
            })
        if "reward" in c or "quality" in c or "evaluate" in c:
            return json.dumps({
                "score": 0.2,
                "reasoning": "Moderate trade quality given current conditions.",
            })
        return json.dumps({"response": "mock response", "value": 0.5})

    def _mock_text_response(self, content: str) -> str:
        return "This is a mock LLM response for testing purposes."


# =====================================================================
# Factory
# =====================================================================

def get_provider(
    backend: str = "auto",
    model_name: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """
    Create an LLM provider with automatic fallback.

    backend:
        "ollama"      -- use local Ollama server
        "huggingface" -- use transformers library
        "mock"        -- deterministic mock (always works)
        "auto"        -- try ollama -> huggingface -> mock
    """
    if backend == "mock":
        return MockProvider(model_name=model_name or "mock")

    if backend == "ollama":
        return OllamaProvider(model_name=model_name or "qwen2.5", **kwargs)

    if backend == "huggingface":
        return HuggingFaceProvider(
            model_name=model_name or "Qwen/Qwen2.5-3B-Instruct", **kwargs
        )

    if backend == "auto":
        # Try Ollama first (fastest, easiest)
        ollama_prov = OllamaProvider(model_name=model_name or "qwen2.5")
        if ollama_prov.is_available():
            logger.info(f"Using Ollama with {ollama_prov.model_name}")
            return ollama_prov

        # Try HuggingFace
        hf_prov = HuggingFaceProvider(
            model_name=model_name or "Qwen/Qwen2.5-3B-Instruct"
        )
        if hf_prov.is_available():
            logger.info(f"Ollama not available, using HuggingFace: {hf_prov.model_name}")
            return hf_prov

        # Fall back to mock
        logger.warning("No LLM backend available. Using MockProvider.")
        return MockProvider(model_name="mock-fallback")

    raise ValueError(f"Unknown backend: {backend}")


# =====================================================================
# Helpers
# =====================================================================

def _parse_json_response(raw: str) -> Dict:
    """Extract and parse JSON from an LLM response, handling markdown fences."""
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find any JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning(f"Failed to parse JSON from LLM response: {raw[:200]}")
        return {"error": "parse_failed", "raw": raw[:500]}
