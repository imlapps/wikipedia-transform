from typing import Annotated

from pydantic import BaseModel, Field

from wikipedia_transform.models.types import OpenAiModelName


class OpenAiModel(BaseModel):
    """Pydantic Model to hold the parameters of an OpenAI model"""

    name: OpenAiModelName = OpenAiModelName.GPT_4O
    temperature: Annotated[float, Field(le=1.0, ge=0.0)] = 0.0
