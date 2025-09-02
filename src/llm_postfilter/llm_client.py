"""
LLM Clients for Security Smell Post-Filtering

This module provides provider-agnostic interfaces and concrete clients for evaluating
GLITCH detections using multiple LLM providers (OpenAI, Ollama, Anthropic, Grok/xAI, and
OpenAI-compatible endpoints).
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import requests

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None


logger = logging.getLogger(__name__)


class LLMDecision(Enum):
    YES = "YES"
    NO = "NO"
    UNCLEAR = "UNCLEAR"
    ERROR = "ERROR"


@dataclass
class LLMResponse:
    decision: LLMDecision
    raw_response: str
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    tokens_used: Optional[int] = None


class Provider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"  # Generic HTTP client for OpenAI-style APIs
    GROK = "grok"  # xAI Grok via OpenAI-compatible API


SYSTEM_PROMPT = (
    "You are an expert security analyst. Answer only YES or NO based on the provided criteria."
)


def _parse_decision_from_text(response_text: str) -> LLMDecision:
    text = (response_text or "").strip().upper()
    if text == "YES":
        return LLMDecision.YES
    if text == "NO":
        return LLMDecision.NO
    if "YES" in text and "NO" not in text:
        return LLMDecision.YES
    if "NO" in text and "YES" not in text:
        return LLMDecision.NO
    logger.warning(f"Unclear LLM response: {response_text}")
    return LLMDecision.UNCLEAR


class BaseLLMClient:
    def __init__(self):
        self.model: str = ""
        self.max_retries: int = 3
        self.retry_delay: float = 1.0
        self.requests_per_minute: int = 60
        self.last_request_time: float = 0.0

    def _enforce_rate_limit(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        min_interval = 60.0 / float(self.requests_per_minute)
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def evaluate_detection(self, prompt: str, max_tokens: int = 50) -> LLMResponse:
        raise NotImplementedError

    def batch_evaluate(self, prompts: List[str], progress_callback=None) -> List[LLMResponse]:
        responses: List[LLMResponse] = []
        total = len(prompts)
        logger.info(f"Starting batch evaluation of {total} prompts")
        for i, p in enumerate(prompts):
            responses.append(self.evaluate_detection(p))
            if progress_callback:
                progress_callback(i + 1, total, responses[-1])
            if (i + 1) % 10 == 0 or (i + 1) == total:
                yes_count = sum(1 for r in responses if r.decision == LLMDecision.YES)
                no_count = sum(1 for r in responses if r.decision == LLMDecision.NO)
                err_count = sum(1 for r in responses if r.decision == LLMDecision.ERROR)
                logger.info(f"Progress: {i + 1}/{total} | YES: {yes_count}, NO: {no_count}, ERROR: {err_count}")
        return responses

    def get_statistics(self, responses: List[LLMResponse]) -> Dict:
        if not responses:
            return {}
        counts = {LLMDecision.YES: 0, LLMDecision.NO: 0, LLMDecision.UNCLEAR: 0, LLMDecision.ERROR: 0}
        total_tokens = 0
        total_time = 0.0
        success = 0
        for r in responses:
            counts[r.decision] += 1
            total_tokens += (r.tokens_used or 0)
            total_time += (r.processing_time or 0.0)
            if r.decision not in (LLMDecision.ERROR, LLMDecision.UNCLEAR):
                success += 1
        return {
            "total_requests": len(responses),
            "yes_decisions": counts[LLMDecision.YES],
            "no_decisions": counts[LLMDecision.NO],
            "unclear_decisions": counts[LLMDecision.UNCLEAR],
            "error_decisions": counts[LLMDecision.ERROR],
            "success_rate": success / len(responses),
            "total_tokens": total_tokens,
            "total_time_seconds": total_time,
            "average_time_per_request": total_time / len(responses) if responses else 0.0,
            "estimated_cost_usd": self._estimate_cost(total_tokens),
        }

    def _estimate_cost(self, total_tokens: int) -> float:
        # Default to GPT-4o mini pricing as a conservative low-cost estimate
        cost_per_1k = 0.0003
        return (total_tokens / 1000.0) * cost_per_1k


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        super().__init__()
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key")
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI client with model: {model}")
    
    def evaluate_detection(self, prompt: str, max_tokens: int = 50) -> LLMResponse:
        start = time.time()
        for attempt in range(self.max_retries):
            try:
                self._enforce_rate_limit()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                content = resp.choices[0].message.content.strip()
                decision = _parse_decision_from_text(content)
                tokens_used = getattr(resp, "usage", None).total_tokens if getattr(resp, "usage", None) else None
                return LLMResponse(
                    decision=decision,
                    raw_response=content,
                    processing_time=time.time() - start,
                    tokens_used=tokens_used,
                )
            except Exception as e:
                logger.warning(f"OpenAI call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    return LLMResponse(LLMDecision.ERROR, "", processing_time=time.time() - start, error_message=str(e))


class OllamaClient(BaseLLMClient):
    def __init__(self, model: str = "codellama:7b", base_url: Optional[str] = None):
        super().__init__()
        self.model = model
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.api_url = f"{self.base_url}/api/generate"
        logger.info(f"Initialized Ollama client with model: {model} @ {self.base_url}")

    def evaluate_detection(self, prompt: str, max_tokens: int = 50) -> LLMResponse:
        start = time.time()
        for attempt in range(self.max_retries):
            try:
                self._enforce_rate_limit()
                full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
                payload = {
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "num_predict": max_tokens,
                    },
                }
                r = requests.post(self.api_url, json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
                content = (data.get("response") or "").strip()
                decision = _parse_decision_from_text(content)
                tokens_used = (data.get("prompt_eval_count", 0) or 0) + (data.get("eval_count", 0) or 0)
                return LLMResponse(
                    decision=decision,
                    raw_response=content,
                    processing_time=time.time() - start,
                    tokens_used=tokens_used,
                )
            except Exception as e:
                logger.warning(f"Ollama call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    return LLMResponse(LLMDecision.ERROR, "", processing_time=time.time() - start, error_message=str(e))


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-latest"):
        super().__init__()
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key")
        self.model = model
        self.client = Anthropic(api_key=self.api_key)
        logger.info(f"Initialized Anthropic client with model: {model}")

    def evaluate_detection(self, prompt: str, max_tokens: int = 50) -> LLMResponse:
        start = time.time()
        for attempt in range(self.max_retries):
            try:
                self._enforce_rate_limit()
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                # Anthropic returns a list of content blocks
                content_parts = msg.content or []
                text = "".join([getattr(part, "text", "") for part in content_parts]).strip()
                decision = _parse_decision_from_text(text)
                usage = getattr(msg, "usage", None)
                total_tokens = None
                if usage and hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
                    total_tokens = int(usage.input_tokens) + int(usage.output_tokens)
                return LLMResponse(
                    decision=decision,
                    raw_response=text,
                    processing_time=time.time() - start,
                    tokens_used=total_tokens,
                )
            except Exception as e:
                logger.warning(f"Anthropic call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    return LLMResponse(LLMDecision.ERROR, "", processing_time=time.time() - start, error_message=str(e))


class OpenAICompatibleClient(BaseLLMClient):
    """Generic client for OpenAI-compatible HTTP APIs (e.g., vLLM, OpenRouter, third-party providers).

    Supports custom headers via environment variables for providers like OpenRouter:
      - OPENROUTER_REFERRER → sets Referer header
      - OPENROUTER_TITLE → sets X-Title header
      - OPENAI_COMPATIBLE_HEADERS → JSON object of extra headers to merge
    """

    def __init__(self, base_url: str, api_key: Optional[str], model: str, extra_headers: Optional[Dict[str, str]] = None):
        super().__init__()
        if not base_url:
            raise ValueError("base_url is required for OpenAI-compatible client")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        self.model = model
        # Prepare optional headers (for OpenRouter and others)
        headers_from_env: Dict[str, str] = {}
        referer = os.getenv("OPENROUTER_REFERRER") or os.getenv("OPENROUTER_REFERER")
        if referer:
            headers_from_env["Referer"] = referer
        title = os.getenv("OPENROUTER_TITLE")
        if title:
            headers_from_env["X-Title"] = title
        try:
            extra_json = os.getenv("OPENAI_COMPATIBLE_HEADERS")
            if extra_json:
                headers_from_env.update(json.loads(extra_json))
        except Exception:
            logger.warning("Failed to parse OPENAI_COMPATIBLE_HEADERS; expected JSON object")
        # Allow direct injection via constructor to override env
        self.extra_headers = {**headers_from_env, **(extra_headers or {})}
        logger.info(f"Initialized OpenAI-compatible client with model: {model} @ {self.base_url}")

    def evaluate_detection(self, prompt: str, max_tokens: int = 50) -> LLMResponse:
        start = time.time()
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # Merge any extra headers (e.g., OpenRouter's Referer/X-Title)
        if self.extra_headers:
            headers.update(self.extra_headers)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        for attempt in range(self.max_retries):
            try:
                self._enforce_rate_limit()
                r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
                r.raise_for_status()
                data = r.json()
                content = (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
                decision = _parse_decision_from_text(content)
                usage = data.get("usage") or {}
                tokens = (usage.get("total_tokens") or 0)
                return LLMResponse(decision=decision, raw_response=content, processing_time=time.time() - start, tokens_used=tokens)
            except Exception as e:
                logger.warning(f"OpenAI-compatible call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    return LLMResponse(LLMDecision.ERROR, "", processing_time=time.time() - start, error_message=str(e))


class GrokClient(OpenAICompatibleClient):
    """Client for xAI Grok models via OpenAI-compatible chat completions API.

    Defaults:
      - base_url: https://api.x.ai/v1
      - model: grok-2-latest
      - api key env: XAI_API_KEY (fallback: GROK_API_KEY)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "grok-2-latest", base_url: Optional[str] = None):
        # Accept overrides via args or env; ensure trailing /v1 present
        raw_base = base_url or os.getenv("GROK_BASE_URL") or os.getenv("XAI_BASE_URL") or "https://api.x.ai"
        base_with_version = raw_base.rstrip("/")
        if not base_with_version.endswith("/v1"):
            base_with_version = base_with_version + "/v1"
        xai_key = api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        super().__init__(base_url=base_with_version, api_key=xai_key, model=model)
        self.model = model
        logger.info(f"Initialized Grok (xAI) client with model: {model} @ {base_with_version}")


def create_llm_client(
    provider: str = Provider.OPENAI.value,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BaseLLMClient:
    prov = Provider(provider)
    if prov == Provider.OPENAI:
        return OpenAIClient(api_key=api_key, model=model)
    if prov == Provider.OLLAMA:
        return OllamaClient(model=model, base_url=base_url)
    if prov == Provider.ANTHROPIC:
        return AnthropicClient(api_key=api_key, model=model)
    if prov == Provider.GROK:
        return GrokClient(api_key=api_key, model=model, base_url=base_url)
    if prov == Provider.OPENAI_COMPATIBLE:
        if not base_url:
            raise ValueError("base_url is required for openai_compatible provider")
        return OpenAICompatibleClient(base_url=base_url, api_key=api_key, model=model)
    raise ValueError(f"Unsupported provider: {provider}")


# Backward-compatibility alias
GPT4OMiniClient = OpenAIClient


def main():
    try:
        client = OpenAIClient()
        test_prompt = (
            "You are an expert in Infrastructure-as-Code security analysis.\n\n"
            "Given this code snippet, is this a hard-coded secret? Answer YES or NO only.\n\n"
            "Code:\napi_key = \"sk-1234567890abcdef\"\n\nAnswer:"
        )
        print("Testing LLM client with simple prompt...")
        response = client.evaluate_detection(test_prompt)
        print(f"Decision: {response.decision}")
        print(f"Raw response: {response.raw_response}")
        print(f"Processing time: {response.processing_time:.2f}s")
    except Exception as e:
        print(f"Test failed (likely missing API key): {e}")


if __name__ == "__main__":
    main()