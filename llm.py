"""Central Claude client factory + shared content helper.

One place to build the LLM, so the resilience policy (timeout, retries) lives in
config and isn't copy-pasted across classifier/responder/knowledge/evaluation. Each
caller still chooses its own max_tokens/temperature — those are task-specific.

`extract_text` normalizes Anthropic message content (a string, or a list of typed
blocks) down to plain text; it was duplicated in two modules before.
"""

from langchain_anthropic import ChatAnthropic

import config


def chat_model(max_tokens: int, temperature: float = 0) -> ChatAnthropic:
    """Construct a Claude client with the centralized timeout/retry policy."""
    return ChatAnthropic(
        model=config.LLM_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=config.LLM_TIMEOUT,
        max_retries=config.LLM_MAX_RETRIES,
    )


def extract_text(content) -> str:
    """Pull plain text out of Anthropic message content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return ""
