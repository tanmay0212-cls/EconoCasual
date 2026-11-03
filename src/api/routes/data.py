"""Dataset validation routes."""

import logging
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, status, UploadFile, File, Form
from src.api.services.data_service import DataService
from src.api.schemas.data import DatasetValidationResponse

logger = logging.getLogger("econocausal.routes.data")
router = APIRouter(prefix="/data", tags=["Dataset Validation"])


@router.post(
    "/validate",
    response_model=DatasetValidationResponse,
    summary="Validate dataset schema, domains, and causal rules",
    description=(
        "Accepts either an uploaded CSV file (via multipart/form-data) "
        "or an approved server-side dataset reference (via application/json or form field). "
        "Arbitrary server file paths are strictly rejected."
    ),
)
async def validate_dataset(request: Request) -> DatasetValidationResponse:
    """Validate customer campaign dataset against schema, domain, and causal integrity rules."""
    content_type = request.headers.get("content-type", "")

    # 1. JSON Request: {"dataset_name": "optimization_retail_data"}
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed JSON body: {str(exc)}",
            )

        ref_name = body.get("dataset_name")
        if not ref_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required field 'dataset_name' in JSON payload.",
            )

        df = DataService.resolve_dataset_reference(ref_name)
        return DataService.validate_dataframe(df, dataset_name=ref_name)

    # 2. Multipart/form-data request: file upload or form dataset_name
    elif "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file")
        form_ref = form.get("dataset_name")

        if uploaded_file is not None and hasattr(uploaded_file, "filename") and uploaded_file.filename:
            df = await DataService.read_uploaded_csv(uploaded_file)
            return DataService.validate_dataframe(df, dataset_name=uploaded_file.filename)
        elif form_ref:
            df = DataService.resolve_dataset_reference(str(form_ref))
            return DataService.validate_dataframe(df, dataset_name=str(form_ref))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Multipart request must include either a 'file' (.csv) or 'dataset_name' form field.",
            )

    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported Content-Type. Please use 'application/json' or 'multipart/form-data'.",
        )
