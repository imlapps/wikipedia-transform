from typing import Annotated

from pydantic import Field

# Tiny type for the limit on retrieved Documents.
DocumentsLimit = Annotated[int, Field(default=1, ge=1)]
