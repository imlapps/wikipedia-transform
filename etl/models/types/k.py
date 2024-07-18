from typing import Annotated

from pydantic import Field

# Tiny type for K, the number of retrieved documents.
K = Annotated[int, Field(default=1, json_schema_extra={"min": 1})]
