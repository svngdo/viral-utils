class LLMError(Exception):
    """Base for all LLM errors."""


class RateLimitError(LLMError):
    """Provider is rate-limited - try next, maybe retry later."""


class AuthError(LLMError):
    """Bad or missing API key - no point retrying this provider."""


class EmptyResponseError(LLMError):
    """Provider returned a result but content was None/empty."""


def normalize_error(e: Exception) -> LLMError:
    """Map provider-specific exceptions to our own error types."""
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    msg = str(e)

    if code in (429,) or "rate limit" in msg.lower():
        error = RateLimitError(msg)
    elif code in (401, 403) or "api key" in msg.lower():
        error = AuthError(msg)
    else:
        error = LLMError(msg)

    error.__cause__ = e  # manually atach the chain
    return error
