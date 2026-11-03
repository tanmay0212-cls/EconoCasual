"""Common and base Pydantic schemas for the EconoCausal API."""

from typing import Generic, TypeVar, Optional, List, Dict, Any
from pydantic import BaseModel, Field

T = TypeVar("T")


class APIErrorDetail(BaseModel):
    """Detailed error object for client responses."""
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error description")
    field: Optional[str] = Field(None, description="Input field associated with the error")


class APIResponse(BaseModel, Generic[T]):
    """Standardized top-level API envelope."""
    success: bool = Field(True, description="Indicates if request succeeded")
    data: Optional[T] = Field(None, description="Payload data")
    errors: Optional[List[APIErrorDetail]] = Field(None, description="Errors if request failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Request execution metadata")
