# backend/models.py

import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Field # Ensure Field is imported from Pydantic for models
from sqlalchemy import Column, String, Integer, DateTime, Text, Float
from sqlalchemy.orm import declarative_base

# SQLAlchemy Base for database models
Base = declarative_base()

class ChatHistory(Base):
    """SQLAlchemy model for storing chat history records."""
    __tablename__ = 'chat_history'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    model_used = Column(String(100), nullable=False)
    provider = Column(String(50), default="unknown")
    cost = Column(Float, default=0.0)
    response_time = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    agents_used = Column(Text, default="")
    workflow_type = Column(String(50), default="single")
    confidence_score = Column(Float, default=0.0)
    tool_calls = Column(Text, default="[]")
    context_updates_json = Column(Text, default="{}")

    def __repr__(self):
        return f"<ChatHistory(id={self.id}, user_id='{self.user_id}', model='{self.model_used}')>"

    @property
    def get_agents_used_list(self) -> List[str]:
        try:
            return json.loads(self.agents_used) if self.agents_used else []
        except json.JSONDecodeError:
            return []

    @property
    def get_tool_calls_list(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.tool_calls) if self.tool_calls else []
        except json.JSONDecodeError:
            return []

    @property
    def get_context_updates(self) -> Dict[str, Any]:
        try:
            return json.loads(self.context_updates_json) if self.context_updates_json else {}
        except json.JSONDecodeError:
            return {}


class ChatRequest(BaseModel):
    """Pydantic model for incoming chat requests."""
    message: str = Field(..., example="Help me scale my SaaS business from $50K to $500K MRR")
    user_id: str = Field("anonymous", example="user_123")
    task_type: str = Field("auto", example="auto", description="Automatically detect, or 'simple_qa', 'code_generation', 'creative_writing', 'business_strategy'")
    user_tier: str = Field("free", example="pro", description="User subscription tier: 'free', 'pro', 'enterprise'")
    business_context: Optional[Dict[str, Any]] = Field(
        None,
        example={
            "business_type": "SaaS",
            "industry": "Marketing Technology",
            "current_revenue": "$50K MRR",
            "main_challenges": ["High customer acquisition cost", "8% monthly churn"],
            "goals": ["Reach $500K MRR in 18 months", "Reduce churn to 3%"]
        },
        description="Detailed context for elite agents to provide tailored responses."
    )
    force_system_nexus: bool = Field(False, description="If true, attempts to route directly to System Nexus.")
    nexus_tool_target: Optional[str] = Field(None, example="Salesforce CRM Connector", description="Specific tool to target within System Nexus if forcing.")


class ChatResponse(BaseModel):
    """Pydantic model for responses from chat and standard AI calls."""
    success: bool = Field(..., example=True)
    response: str = Field(..., example="Here's a strategy to scale your SaaS...")
    model: str = Field(..., example="Elite Agent Team" ,description="Model or agent team used for the response.")
    provider: str = Field(..., example="Elite Command Center", description="Provider of the model/service.")
    cost: float = Field(0.0, example=0.015, description="Estimated cost of the operation.")
    response_time: float = Field(0.0, example=5.23, description="Time taken for the response in seconds.")
    cached: bool = Field(False, description="True if the response was served from cache (not yet implemented).")
    reasoning: str = Field("", example="Detected 'business_strategy', routed to Elite Agents.")
    agents_used: List[str] = Field([], example=["Growth Assassin", "Data Oracle"])
    workflow_type: str = Field("single", example="sequential", description="Type of workflow executed: 'single', 'sequential', 'parallel', 'hybrid'.")
    confidence_score: float = Field(0.0, example=0.92, description="Confidence score of the response from agents.")
    tool_calls_made: List[Dict[str, Any]] = Field([], description="Details of external tool calls made during the request.")


class FunctionCallRequest(BaseModel):
    """Pydantic model for a direct function/tool call request (e.g., to System Nexus)."""
    function_name: str = Field(..., example="update_crm_record")
    arguments: Dict[str, Any] = Field(..., example={"customer_id": "C123", "status": "Lead", "notes": "Follow up required"})
    user_id: str = Field("anonymous", example="admin_user")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the function call.")


class FunctionCallResponse(BaseModel):
    """Pydantic model for the response of a direct function/tool call."""
    success: bool = Field(..., example=True)
    result: Any = Field(..., example={"status": "success", "record_updated": True})
    function_name: str = Field(..., example="update_crm_record")
    execution_time: float = Field(0.0, example=1.5)
    error: Optional[str] = Field(None, example="Authentication failed.")


class DevelopmentTask(BaseModel):
    """Pydantic model for a development-related task request."""
    task_description: str = Field(..., example="Implement a new user authentication flow.")


# Elite Agent specific Pydantic models for API integration
class EliteAgentRequest(BaseModel):
    """Pydantic model for direct requests to elite agents or the command center."""
    message: str = Field(..., example="Develop a growth strategy for my startup.")
    user_id: str = Field("anonymous", example="api_user_alpha")
    business_type: str = Field("", example="SaaS")
    industry: str = Field("", example="Fintech")
    target_audience: str = Field("", example="Small businesses")
    current_revenue: str = Field("", example="$10K MRR")
    main_challenges: List[str] = Field([], example=["High churn", "Low customer acquisition"])
    goals: List[str] = Field([], example=["Reach $100K MRR", "Improve retention to 90%"])
    call_nexus_tool: Optional[str] = Field(None, example="Google Cloud Integration:trigger_workflow", description="If set, attempts to directly trigger a tool via System Nexus.")
    nexus_tool_args: Optional[Dict[str, Any]] = Field(None, description="Arguments for the System Nexus tool call.")


class EliteAgentResponse(BaseModel):
    """Pydantic model for responses from the Elite Agent Command Center."""
    success: bool = Field(..., example=True)
    routing_decision: Dict[str, Any] = Field(..., description="Decision made by the Intent Classifier.")
    workflow_result: Dict[str, Any] = Field(..., description="Results from the orchestrated agent workflow.")
    execution_time: float = Field(..., example=12.5)
    agents_used: List[str] = Field(..., example=["Growth Assassin", "Data Oracle"])
    error: Optional[str] = Field(None, example="Failed to execute workflow.")
    tool_calls_made: List[Dict[str, Any]] = Field([], description="Details of external tool calls made by System Nexus.")