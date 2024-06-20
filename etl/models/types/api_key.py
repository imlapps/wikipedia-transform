from typing import Annotated

from pydantic import Field

# Tiny type for an OpenAI API key.
ApiKey = Annotated[str, Field(json_schema_extra={"strip_whitespace": True})]
