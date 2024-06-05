from enum import Enum


class OpenAiModelName(str, Enum):
    """An enum of OpenAI model names."""

    GPT_4O = "gpt-4o"
    GPT_4_TURBO = "gpt-4-turbo"
