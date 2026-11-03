"""Schemas for customer-level CATE and counterfactual purchase probability prediction."""

from typing import List, Optional
from pydantic import BaseModel, Field


class CustomerFeatureRecord(BaseModel):
    """Pre-treatment customer profile for causal effect prediction."""
    customer_id: str = Field(..., description="Unique customer identifier", examples=["CUST_001"])
    age: float = Field(..., ge=18, le=100, description="Customer age in years", examples=[35.0])
    loyalty_score: float = Field(..., ge=0, le=100, description="Customer loyalty index [0-100]", examples=[62.5])
    previous_purchases: int = Field(..., ge=0, description="Historical purchase transaction count", examples=[4])
    customer_segment: str = Field(
        ...,
        description="Behavioral customer segment: 'budget_seeker' | 'casual_shopper' | 'regular_buyer' | 'vip_loyalist' | 'Standard'",
        examples=["regular_buyer"]
    )
    product_price: Optional[float] = Field(85.0, ge=0.0, description="Unit base price in USD", examples=[85.0])
    gross_margin_rate: Optional[float] = Field(0.40, ge=0.0, le=1.0, description="Product margin rate [0-1]", examples=[0.40])


class PredictEffectsRequest(BaseModel):
    """Request payload for batch treatment-effect prediction."""
    records: List[CustomerFeatureRecord] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of customer records (max 5,000 per request)"
    )


class CustomerEffectPrediction(BaseModel):
    """Individual customer causal prediction and counterfactual purchase probabilities."""
    customer_id: str
    predicted_effect_10_vs_0: float = Field(..., description="Estimated CATE lift of $10 discount relative to $0")
    predicted_effect_20_vs_0: float = Field(..., description="Estimated CATE lift of $20 discount relative to $0")
    predicted_purchase_probability_0: float = Field(..., ge=0.0, le=1.0, description="P(purchase | $0 discount)")
    predicted_purchase_probability_10: float = Field(..., ge=0.0, le=1.0, description="P(purchase | $10 discount)")
    predicted_purchase_probability_20: float = Field(..., ge=0.0, le=1.0, description="P(purchase | $20 discount)")


class PredictEffectsResponse(BaseModel):
    """Batch treatment-effect prediction response."""
    model_version: str
    estimator_class: str
    record_count: int
    predictions: List[CustomerEffectPrediction]
