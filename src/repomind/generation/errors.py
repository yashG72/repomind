"""Answer generation failures. Messages are safe to show users: they never contain the API
key, the prompt, retrieved source code or the provider's response body."""


class LLMError(Exception):
    """Base class for answer generation failures."""


class LLMNotConfigured(LLMError):
    """No LLM provider (or no API key) is configured, so questions cannot be answered."""


class LLMAuthenticationError(LLMError):
    """The provider rejected the configured credentials or model."""


class LLMRateLimited(LLMError):
    """The provider is rate limiting requests."""


class LLMTimeout(LLMError):
    """The provider did not answer within the configured timeout."""


class LLMUnavailable(LLMError):
    """The provider could not be reached or had an internal error."""


class LLMRefused(LLMError):
    """The provider's safety systems declined to answer."""


class MalformedLLMOutput(LLMError):
    """The model's reply was not in the required format."""
