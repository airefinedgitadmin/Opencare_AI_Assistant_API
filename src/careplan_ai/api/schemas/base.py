"""Shared base class for wire-format schemas.

Every request/response model in `api.schemas` inherits `CamelModel` so the
HTTP contract uses camelCase JSON (matching the Java backend's Jackson
defaults) while the Python code itself stays snake_case.
"""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
