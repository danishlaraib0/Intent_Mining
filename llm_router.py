"""Single-file Universal LLM Router.
Compatible with Groq, OpenAI, Azure OpenAI, Anthropic, Gemini, OpenRouter, DeepSeek, Ollama, and LangChain.
Supports Streaming (stream=True, .stream(), .astream()), reasoning_effort, max_completion_tokens, batching, and embeddings.

Usage:
    # 1. Zero config - reads from .env / environment variables:
    from dataset.llm_router import LLM
    llm = LLM()
    print(llm("Hello, world!"))

    # 2. Groq (with streaming, reasoning_effort, max_completion_tokens):
    llm = LLM(
        provider="groq",
        model="qwen/qwen3.8-27b",
        temperature=0.6,
        top_p=0.95
    )

    # Streaming call:
    for chunk in llm.stream("Write a short story.", reasoning_effort="default", max_completion_tokens=2048):
        print(chunk, end="", flush=True)

    # Direct chat streaming:
    completion = llm.chat(
        [{"role": "user", "content": "Hello"}],
        stream=True,
        max_completion_tokens=2048,
        reasoning_effort="default"
    )
    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")

    # 3. Azure OpenAI:
    llm = LLM(
        provider="azure",
        endpoint="https://your-resource.openai.azure.com",
        api_version="2024-02-01",
        model="gpt-4o",
        api_key="your-key",
    )
"""

from __future__ import annotations
import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence, Union

# Auto-load .env if available
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True), override=False)
except ImportError:
    pass

# OpenAI SDK for direct, Azure, Groq, OpenRouter, DeepSeek, Ollama
try:
    from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

import httpx

logger = logging.getLogger("llm_router")


# ==========================================
# Data Models
# ==========================================

@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> LLMUsage:
        if not data:
            return cls()
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0) or 0,
            completion_tokens=data.get("completion_tokens", 0) or 0,
            total_tokens=data.get("total_tokens", 0) or 0,
        )


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return f"<LLMResponse provider={self.provider} model={self.model} tokens={self.usage.total_tokens}>"


# ==========================================
# Configuration Container
# ==========================================

@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    top_p: float = 1.0
    timeout: float = 60.0
    max_retries: int = 3
    system_prompt: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        top_p: Optional[float] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        system_prompt: Optional[str] = None,
        embedding_model: Optional[str] = None,
        **extra_kwargs: Any,
    ) -> LLMConfig:
        resolved_provider = provider or os.getenv("LLM_PROVIDER") or os.getenv("MODEL_PROVIDER")

        resolved_endpoint = (
            endpoint
            or base_url
            or os.getenv("LLM_ENDPOINT")
            or os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("AZURE_ENDPOINT")
            or os.getenv("OLLAMA_BASE_URL")
            or os.getenv("GROQ_BASE_URL")
        )

        resolved_api_version = (
            api_version
            or os.getenv("LLM_API_VERSION")
            or os.getenv("OPENAI_API_VERSION")
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("AZURE_API_VERSION")
        )

        # Auto-detect provider if missing
        if not resolved_provider:
            if os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_ENDPOINT") or (resolved_endpoint and "azure" in resolved_endpoint.lower()):
                resolved_provider = "azure"
            elif os.getenv("GROQ_API_KEY") or (resolved_endpoint and "groq" in resolved_endpoint.lower()):
                resolved_provider = "groq"
            elif os.getenv("ANTHROPIC_API_KEY") or (model and "claude" in model.lower()):
                resolved_provider = "anthropic"
            elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or (model and "gemini" in model.lower()):
                resolved_provider = "gemini"
            elif os.getenv("OPENROUTER_API_KEY") or (resolved_endpoint and "openrouter" in resolved_endpoint.lower()):
                resolved_provider = "openrouter"
            elif os.getenv("DEEPSEEK_API_KEY") or (resolved_endpoint and "deepseek" in resolved_endpoint.lower()):
                resolved_provider = "deepseek"
            elif os.getenv("OLLAMA_BASE_URL") or (resolved_endpoint and "11434" in resolved_endpoint):
                resolved_provider = "ollama"
            else:
                resolved_provider = "openai"

        resolved_provider = resolved_provider.lower().strip()
        if resolved_provider in ("azure_openai",):
            resolved_provider = "azure"
        elif resolved_provider in ("google",):
            resolved_provider = "gemini"

        # Model defaults
        default_models = {
            "openai": "gpt-4o",
            "azure": "gpt-4o",
            "groq": "llama-3.3-70b-versatile",
            "anthropic": "claude-3-5-sonnet-20241022",
            "gemini": "gemini-1.5-flash",
            "openrouter": "openai/gpt-4o",
            "deepseek": "deepseek-chat",
            "ollama": "llama3",
        }
        resolved_model = (
            model
            or os.getenv("LLM_MODEL")
            or os.getenv("MODEL_NAME")
            or (os.getenv("GROQ_MODEL") if resolved_provider == "groq" else None)
            or (os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") if resolved_provider == "azure" else None)
            or (os.getenv("OPENAI_MODEL") if resolved_provider == "openai" else None)
            or default_models.get(resolved_provider, "gpt-4o")
        )

        # API Key defaults
        resolved_api_key = api_key or os.getenv("LLM_API_KEY")
        if not resolved_api_key:
            if resolved_provider == "groq":
                resolved_api_key = os.getenv("GROQ_API_KEY")
            elif resolved_provider == "azure":
                resolved_api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY") or os.getenv("OPENAI_API_KEY")
            elif resolved_provider == "openai":
                resolved_api_key = os.getenv("OPENAI_API_KEY")
            elif resolved_provider == "anthropic":
                resolved_api_key = os.getenv("ANTHROPIC_API_KEY")
            elif resolved_provider == "gemini":
                resolved_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            elif resolved_provider == "openrouter":
                resolved_api_key = os.getenv("OPENROUTER_API_KEY")
            elif resolved_provider == "deepseek":
                resolved_api_key = os.getenv("DEEPSEEK_API_KEY")
            elif resolved_provider == "ollama":
                resolved_api_key = os.getenv("OLLAMA_API_KEY", "ollama")

        # Endpoint defaults
        default_endpoints = {
            "groq": "https://api.groq.com/openai/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "ollama": "http://localhost:11434/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
        }
        if not resolved_endpoint and resolved_provider in default_endpoints:
            resolved_endpoint = default_endpoints[resolved_provider]

        # Numeric and extra params
        resolved_temp = float(temperature if temperature is not None else os.getenv("LLM_TEMPERATURE", "0.0"))
        env_max = os.getenv("LLM_MAX_TOKENS")
        resolved_max_tokens = int(max_tokens) if max_tokens is not None else (int(env_max) if env_max else None)
        resolved_max_comp_tokens = int(max_completion_tokens) if max_completion_tokens is not None else (
            int(os.getenv("LLM_MAX_COMPLETION_TOKENS")) if os.getenv("LLM_MAX_COMPLETION_TOKENS") else None
        )
        resolved_reasoning = reasoning_effort or os.getenv("LLM_REASONING_EFFORT")
        resolved_top_p = float(top_p if top_p is not None else os.getenv("LLM_TOP_P", "1.0"))
        resolved_timeout = float(timeout or os.getenv("LLM_TIMEOUT", "60.0"))
        resolved_retries = int(max_retries or os.getenv("LLM_MAX_RETRIES", "3"))
        resolved_emb = embedding_model or os.getenv("LLM_EMBEDDING_MODEL") or "text-embedding-3-small"

        return cls(
            provider=resolved_provider,
            model=resolved_model,
            api_key=resolved_api_key,
            endpoint=resolved_endpoint,
            base_url=resolved_endpoint,
            api_version=resolved_api_version,
            temperature=resolved_temp,
            max_tokens=resolved_max_tokens,
            max_completion_tokens=resolved_max_comp_tokens,
            reasoning_effort=resolved_reasoning,
            top_p=resolved_top_p,
            timeout=resolved_timeout,
            max_retries=resolved_retries,
            system_prompt=system_prompt or os.getenv("LLM_SYSTEM_PROMPT"),
            embedding_model=resolved_emb,
            extra_params=extra_kwargs,
        )


# ==========================================
# Main Unified LLMRouter / LLM Class
# ==========================================

class LLMRouter:
    """Universal Single-File LLM Router with full streaming and Groq/OpenAI support."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        top_p: Optional[float] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        system_prompt: Optional[str] = None,
        embedding_model: Optional[str] = None,
        **extra_kwargs: Any,
    ):
        self.config = LLMConfig.from_env(
            provider=provider,
            model=model,
            api_key=api_key,
            endpoint=endpoint,
            base_url=base_url,
            api_version=api_version,
            temperature=temperature,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            top_p=top_p,
            timeout=timeout,
            max_retries=max_retries,
            system_prompt=system_prompt,
            embedding_model=embedding_model,
            **extra_kwargs,
        )
        self._init_clients()

    def _init_clients(self) -> None:
        provider = self.config.provider.lower()

        if provider == "azure":
            if not HAS_OPENAI:
                raise ImportError("Run `pip install openai` for Azure OpenAI.")
            endpoint = self.config.endpoint or self.config.base_url
            if not endpoint:
                raise ValueError("Azure OpenAI requires AZURE_OPENAI_ENDPOINT or endpoint='...'.")
            api_ver = self.config.api_version or "2024-02-01"
            self._sync = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=self.config.api_key,
                api_version=api_ver,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            self._async = AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                api_key=self.config.api_key,
                api_version=api_ver,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )

        elif provider in ("groq", "openai", "openrouter", "deepseek", "ollama", "gemini", "custom"):
            if not HAS_OPENAI:
                raise ImportError("Run `pip install openai`.")
            base_url = self.config.endpoint or self.config.base_url
            api_key = self.config.api_key or "placeholder-key"
            self._sync = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            self._async = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )

        elif provider == "anthropic":
            self._http_sync = httpx.Client(timeout=self.config.timeout)
            self._http_async = httpx.AsyncClient(timeout=self.config.timeout)

    def _format_msgs(
        self,
        messages: Union[str, List[Union[Dict[str, Any], ChatMessage]]],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        sys = system_prompt or self.config.system_prompt
        if sys:
            formatted.append({"role": "system", "content": sys})

        if isinstance(messages, str):
            formatted.append({"role": "user", "content": messages})
            return formatted

        for m in messages:
            if isinstance(m, ChatMessage):
                formatted.append(m.to_dict())
            elif isinstance(m, dict):
                formatted.append(m)
            else:
                formatted.append({"role": "user", "content": str(m)})
        return formatted

    # --- Callable & Text Generation ---

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        resp = self.chat(prompt, **kwargs)
        return resp.content if isinstance(resp, LLMResponse) else str(resp)

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        resp = await self.achat(prompt, **kwargs)
        return resp.content if isinstance(resp, LLMResponse) else str(resp)

    # --- Chat & Streaming Methods ---

    def chat(
        self,
        messages: Union[str, List[Union[Dict[str, Any], ChatMessage]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[LLMResponse, Any]:
        """Send chat messages. If stream=True, returns standard chunk iterator."""
        msgs = self._format_msgs(messages, system_prompt)
        use_model = model or self.config.model
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        comp_tokens = max_completion_tokens if max_completion_tokens is not None else self.config.max_completion_tokens
        effort = reasoning_effort or self.config.reasoning_effort
        p = top_p if top_p is not None else self.config.top_p

        if self.config.provider == "anthropic":
            return self._anthropic_chat_sync(msgs, use_model, temp, tokens, stream=stream, **kwargs)

        req_params: Dict[str, Any] = {
            "model": use_model,
            "messages": msgs,
            "temperature": temp,
            "top_p": p,
            "stream": stream,
            **self.config.extra_params,
            **kwargs,
        }
        if tokens is not None:
            req_params["max_tokens"] = tokens
        if comp_tokens is not None:
            req_params["max_completion_tokens"] = comp_tokens
        if effort is not None:
            req_params["reasoning_effort"] = effort

        res = self._sync.chat.completions.create(**req_params)

        if stream:
            return res  # Generator of chunks (chunk.choices[0].delta.content)

        choice = res.choices[0]
        usage = LLMUsage.from_dict(res.usage.model_dump() if hasattr(res, "usage") and res.usage else None)

        return LLMResponse(
            content=choice.message.content or "",
            model=res.model or use_model,
            provider=self.config.provider,
            usage=usage,
            finish_reason=choice.finish_reason,
            raw_response=res.model_dump() if hasattr(res, "model_dump") else None,
        )

    async def achat(
        self,
        messages: Union[str, List[Union[Dict[str, Any], ChatMessage]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[LLMResponse, Any]:
        """Asynchronously send chat messages."""
        msgs = self._format_msgs(messages, system_prompt)
        use_model = model or self.config.model
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        comp_tokens = max_completion_tokens if max_completion_tokens is not None else self.config.max_completion_tokens
        effort = reasoning_effort or self.config.reasoning_effort
        p = top_p if top_p is not None else self.config.top_p

        if self.config.provider == "anthropic":
            return await self._anthropic_chat_async(msgs, use_model, temp, tokens, stream=stream, **kwargs)

        req_params: Dict[str, Any] = {
            "model": use_model,
            "messages": msgs,
            "temperature": temp,
            "top_p": p,
            "stream": stream,
            **self.config.extra_params,
            **kwargs,
        }
        if tokens is not None:
            req_params["max_tokens"] = tokens
        if comp_tokens is not None:
            req_params["max_completion_tokens"] = comp_tokens
        if effort is not None:
            req_params["reasoning_effort"] = effort

        res = await self._async.chat.completions.create(**req_params)

        if stream:
            return res

        choice = res.choices[0]
        usage = LLMUsage.from_dict(res.usage.model_dump() if hasattr(res, "usage") and res.usage else None)

        return LLMResponse(
            content=choice.message.content or "",
            model=res.model or use_model,
            provider=self.config.provider,
            usage=usage,
            finish_reason=choice.finish_reason,
            raw_response=res.model_dump() if hasattr(res, "model_dump") else None,
        )

    # --- Clean Streaming Helpers ---

    def stream(
        self,
        prompt: Union[str, List[Union[Dict[str, Any], ChatMessage]]],
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Convenient generator that yields text chunks directly."""
        stream_response = self.chat(prompt, stream=True, **kwargs)
        for chunk in stream_response:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", "") or ""
                if text:
                    yield text

    async def astream(
        self,
        prompt: Union[str, List[Union[Dict[str, Any], ChatMessage]]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Convenient async generator that yields text chunks directly."""
        stream_response = await self.achat(prompt, stream=True, **kwargs)
        async for chunk in stream_response:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", "") or ""
                if text:
                    yield text

    # --- Batching ---

    def batch(self, prompts: Sequence[str], concurrency: int = 10, **kwargs: Any) -> List[str]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                pass

        return loop.run_until_complete(self.abatch(prompts, concurrency=concurrency, **kwargs))

    async def abatch(self, prompts: Sequence[str], concurrency: int = 10, **kwargs: Any) -> List[str]:
        semaphore = asyncio.Semaphore(concurrency)

        async def _worker(p: str) -> str:
            async with semaphore:
                return await self.agenerate(p, **kwargs)

        return await asyncio.gather(*[_worker(p) for p in prompts])

    # --- Embeddings ---

    def embed(self, texts: Union[str, List[str]], model: Optional[str] = None, **kwargs: Any) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        use_model = model or self.config.embedding_model
        res = self._sync.embeddings.create(input=texts, model=use_model, **kwargs)
        return [item.embedding for item in res.data]

    async def aembed(self, texts: Union[str, List[str]], model: Optional[str] = None, **kwargs: Any) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        use_model = model or self.config.embedding_model
        res = await self._async.embeddings.create(input=texts, model=use_model, **kwargs)
        return [item.embedding for item in res.data]

    # --- LangChain Interoperability ---

    def to_langchain(self) -> Any:
        provider = self.config.provider.lower()
        if provider == "azure":
            from langchain_openai import AzureChatOpenAI
            return AzureChatOpenAI(
                azure_endpoint=self.config.endpoint or self.config.base_url,
                azure_deployment=self.config.model,
                api_key=self.config.api_key,
                api_version=self.config.api_version or "2024-02-01",
                temperature=self.config.temperature,
            )
        else:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.config.model,
                api_key=self.config.api_key or "sk-placeholder",
                base_url=self.config.endpoint or self.config.base_url,
                temperature=self.config.temperature,
            )

    def invoke(self, input_data: Any, **kwargs: Any) -> Any:
        if isinstance(input_data, str):
            return self.generate(input_data, **kwargs)
        elif isinstance(input_data, list):
            return self.chat(input_data, **kwargs)
        elif isinstance(input_data, dict):
            content = input_data.get("input") or input_data.get("prompt") or str(input_data)
            return self.generate(content, **kwargs)
        return self.generate(str(input_data), **kwargs)

    async def ainvoke(self, input_data: Any, **kwargs: Any) -> Any:
        if isinstance(input_data, str):
            return await self.agenerate(input_data, **kwargs)
        elif isinstance(input_data, list):
            return await self.achat(input_data, **kwargs)
        elif isinstance(input_data, dict):
            content = input_data.get("input") or input_data.get("prompt") or str(input_data)
            return await self.agenerate(content, **kwargs)
        return await self.agenerate(str(input_data), **kwargs)

    def _anthropic_chat_sync(self, msgs: List[Dict[str, Any]], model: str, temp: float, tokens: Optional[int], **kwargs: Any) -> LLMResponse:
        url = self.config.endpoint or "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": self.config.api_key or "", "anthropic-version": self.config.api_version or "2023-06-01", "content-type": "application/json"}
        user_msgs = [m for m in msgs if m["role"] != "system"]
        sys_msgs = " ".join([m["content"] for m in msgs if m["role"] == "system"])
        payload = {"model": model, "messages": user_msgs, "max_tokens": tokens or 4096, "temperature": temp, **kwargs}
        if sys_msgs:
            payload["system"] = sys_msgs
        resp = self._http_sync.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = "".join([b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"])
        return LLMResponse(content=content, model=data.get("model", model), provider="anthropic", raw_response=data)

    async def _anthropic_chat_async(self, msgs: List[Dict[str, Any]], model: str, temp: float, tokens: Optional[int], **kwargs: Any) -> LLMResponse:
        url = self.config.endpoint or "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": self.config.api_key or "", "anthropic-version": self.config.api_version or "2023-06-01", "content-type": "application/json"}
        user_msgs = [m for m in msgs if m["role"] != "system"]
        sys_msgs = " ".join([m["content"] for m in msgs if m["role"] == "system"])
        payload = {"model": model, "messages": user_msgs, "max_tokens": tokens or 4096, "temperature": temp, **kwargs}
        if sys_msgs:
            payload["system"] = sys_msgs
        resp = await self._http_async.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = "".join([b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"])
        return LLMResponse(content=content, model=data.get("model", model), provider="anthropic", raw_response=data)

    def __repr__(self) -> str:
        return f"<LLMRouter provider='{self.config.provider}' model='{self.config.model}' endpoint='{self.config.endpoint or 'default'}' temp={self.config.temperature}>"


# Convenient Aliases & Export
LLM = LLMRouter


def get_llm(**kwargs: Any) -> LLMRouter:
    return LLMRouter(**kwargs)
