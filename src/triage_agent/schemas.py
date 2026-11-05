"""Typed inputs and the structured decision card every model must return."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal[
    "billing", "account_access", "technical_issue", "feature_request", "cancellation", "security", "privacy"
]
Priority = Literal["low", "medium", "high", "urgent"]
Action = Literal["auto_resolve", "route_to_team", "escalate_to_human"]
Team = Literal["billing", "engineering", "product", "retention", "security", "privacy"]


class Ticket(BaseModel):
    id: str
    subject: str
    body: str
    customer_id: str | None = None


class TriageDecision(BaseModel):
    """The triage decision card for one support ticket."""

    category: Category
    priority: Priority
    action: Action
    team: Team | None = Field(default=None, description="Owning team; required when action is route_to_team")
    confidence: float = Field(ge=0.0, le=1.0, description="How sure you are this decision follows the policies (0-1)")
    rationale: str = Field(description="One or two sentences naming the policy and the evidence used")
    policies_applied: list[str] = Field(default_factory=list, description="Policy topics this decision relies on")
    customer_reply: str | None = Field(default=None, description="Short reply to send when action is auto_resolve")
