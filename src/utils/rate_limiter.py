"""Rate-limited LLM caller with automatic fallback and per-call logging."""

import time
from typing import Any, Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llm_router import LLM, LLMResponse
from src.utils.logger import PipelineLogger


class RateLimitedLLM:
    """Wraps llm_router.LLM with rate limiting, retry, fallback, and logging."""

    def __init__(self, cfg: Dict[str, Any], logger: PipelineLogger):
        llm_cfg = cfg["llm"]
        rl = llm_cfg["rate_limits"]

        self.primary = LLM(
            provider=llm_cfg["primary"]["provider"],
            model=llm_cfg["primary"]["model"],
            top_p=llm_cfg["primary"]["top_p"],
            max_completion_tokens=llm_cfg["primary"]["max_completion_tokens"],
            reasoning_effort=llm_cfg["primary"].get("reasoning_effort"),
        )
        self.fallback = LLM(
            provider=llm_cfg["fallback"]["provider"],
            model=llm_cfg["fallback"]["model"],
            top_p=llm_cfg["fallback"]["top_p"],
            max_completion_tokens=llm_cfg["fallback"]["max_completion_tokens"],
        )

        self.delay_between_requests = rl["delay_between_requests"]
        self.delay_between_batches = rl["delay_between_batches"]
        self.max_retries = rl["max_retries"]
        self.retry_delay = rl["retry_delay"]
        self.logger = logger

        self._last_request_time = 0.0

    def _throttle(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.delay_between_requests:
            time.sleep(self.delay_between_requests - elapsed)
        self._last_request_time = time.time()

    def call(
        self,
        prompt: str,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        stage: str = "unknown",
    ) -> str:
        """Call primary, fall back on failure. Logs every call."""

        # Try primary
        for attempt in range(self.max_retries + 1):
            self._throttle()
            t0 = time.time()
            try:
                resp = self.primary.chat(prompt, temperature=temperature, system_prompt=system_prompt)
                latency = (time.time() - t0) * 1000
                usage = resp.usage if hasattr(resp, "usage") else None

                self.logger.log_call(
                    stage=stage,
                    model=self.primary.config.model,
                    is_fallback=False,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    latency_ms=latency,
                    success=True,
                    input_preview=prompt[:80],
                    output_preview=resp.content[:120],
                )
                return resp.content

            except Exception as e:
                latency = (time.time() - t0) * 1000
                err = str(e).lower()

                self.logger.log_call(
                    stage=stage,
                    model=self.primary.config.model,
                    is_fallback=False,
                    prompt_tokens=0, completion_tokens=0, total_tokens=0,
                    latency_ms=latency,
                    success=False,
                    error=str(e)[:200],
                    input_preview=prompt[:80],
                )

                if "rate_limit" in err or "429" in err or "too many" in err:
                    wait = self.retry_delay * (attempt + 1)
                    print(f"  ⏳ Rate limit (primary). Waiting {wait}s... ({attempt+1}/{self.max_retries+1})")
                    time.sleep(wait)
                    continue
                else:
                    print(f"  ⚠️  Primary error: {str(e)[:80]}. Trying fallback...")
                    break

        # Fallback
        for attempt in range(self.max_retries + 1):
            self._throttle()
            t0 = time.time()
            try:
                resp = self.fallback.chat(prompt, temperature=temperature, system_prompt=system_prompt)
                latency = (time.time() - t0) * 1000
                usage = resp.usage if hasattr(resp, "usage") else None

                self.logger.log_call(
                    stage=stage,
                    model=self.fallback.config.model,
                    is_fallback=True,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    latency_ms=latency,
                    success=True,
                    input_preview=prompt[:80],
                    output_preview=resp.content[:120],
                )
                return resp.content

            except Exception as e:
                latency = (time.time() - t0) * 1000
                self.logger.log_call(
                    stage=stage,
                    model=self.fallback.config.model,
                    is_fallback=True,
                    prompt_tokens=0, completion_tokens=0, total_tokens=0,
                    latency_ms=latency,
                    success=False,
                    error=str(e)[:200],
                    input_preview=prompt[:80],
                )

                err = str(e).lower()
                if "rate_limit" in err or "429" in err:
                    wait = self.retry_delay * (attempt + 1)
                    print(f"  ⏳ Rate limit (fallback). Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError("All retries exhausted on both primary and fallback models.")

    def batch_delay(self) -> None:
        print(f"  💤 Batch cooldown ({self.delay_between_batches}s)...")
        time.sleep(self.delay_between_batches)
