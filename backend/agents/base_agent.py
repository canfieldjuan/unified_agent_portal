# backend/agents/base_agent.py

import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field # <--- CRITICALLY CORRECTED: Imported 'field' (lowercase)
from enum import Enum
import openai
import json

class AgentStatus(Enum):
    """Represents the current operational status of an AI agent."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class AgentCapability(Enum):
    """Defines specific areas of expertise or functionality for agents."""
    GROWTH_STRATEGY = "growth_strategy"
    CONVERSION_OPTIMIZATION = "conversion_optimization"
    DESIGN_SYSTEMS = "design_systems"
    EMAIL_MARKETING = "email_marketing"
    CODE_OPTIMIZATION = "code_optimization"
    DATA_ANALYSIS = "data_analysis"
    SYSTEM_ARCHITECTURE = "system_architecture"
    EXTERNAL_INTEGRATION = "external_integration"

@dataclass
class AgentContext:
    """
    Data class to hold dynamic business and user context for agents.
    This context evolves as agents process requests and generate new insights.
    """
    business_type: str = ""
    industry: str = ""
    target_audience: str = ""
    current_revenue: str = ""
    main_challenges: List[str] = field(default_factory=list) # <--- CORRECTED: used 'field' (lowercase)
    goals: List[str] = field(default_factory=list) # <--- CORRECTED: used 'field' (lowercase)
    previous_results: Dict[str, Any] = field(default_factory=dict) # <--- CORRECTED: used 'field' (lowercase)
    user_preferences: Dict[str, Any] = field(default_factory=dict) # <--- CORRECTED: used 'field' (lowercase)
    business_evolution_history: List[Dict[str, Any]] = field(default_factory=list) # <--- CORRECTED: used 'field' (lowercase)
    relationship_memory: Dict[str, Any] = field(default_factory=dict) # <--- CORRECTED: used 'field' (lowercase)
    semantic_tags: List[str] = field(default_factory=list) # <--- CORRECTED: used 'field' (lowercase)

@dataclass
class AgentResult:
    """
    Data class to encapsulate the outcome of an agent's execution.
    Provides structured output, performance metrics, and recommendations.
    """
    agent_name: str
    success: bool
    output: Dict[str, Any] # Standardized output, e.g., {"response": "...", "type": "copy"}
    execution_time: float
    cost: float
    confidence_score: float # 0.0 to 1.0, how confident the agent is in its output
    next_recommended_agents: List[str] = field(default_factory=list) # <--- CORRECTED: used 'field' (lowercase)
    context_updates: Dict[str, Any] = field(default_factory=dict) # <--- CORRECTED: used 'field' (lowercase)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list) # <--- CORRECTED: used 'field' (lowercase)

class BaseKillerAgent:
    """
    Abstract base class for all specialized killer AI agents.
    Provides common functionality for execution, metrics, and interaction with LLMs.
    """

    def __init__(self, name: str, capabilities: List[AgentCapability], openai_client: Optional[openai.AsyncOpenAI]):
        self.name = name
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE
        self.client = openai_client
        self.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "success_rate": 1.0,
            "avg_execution_time": 0.0,
            "avg_confidence_score": 0.0,
            "total_cost": 0.0
        }

    async def execute(self, request: str, context: AgentContext) -> AgentResult:
        """
        Abstract method: Executes the agent's specific task.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the execute method.")

    async def _call_openai(self, system_prompt: str, user_message: str, model: str = "gpt-4-turbo-preview") -> tuple[str, float]:
        """
        Internal method to interact with the OpenAI API.
        Returns response content and estimated cost.
        """
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized. Check OPENAI_API_KEY.")

        try:
            response = await self.client.chat.completions.acreate(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            cost_per_token_input = 0.01 / 1000
            cost_per_token_output = 0.03 / 1000
            estimated_cost = (prompt_tokens * cost_per_token_input) + (completion_tokens * cost_per_token_output)
            
            return response.choices[0].message.content, estimated_cost
        except openai.APIConnectionError as e:
            raise RuntimeError(f"Could not connect to OpenAI API: {e}")
        except openai.RateLimitError as e:
            raise RuntimeError(f"OpenAI API rate limit exceeded: {e}")
        except openai.APIStatusError as e:
            raise RuntimeError(f"OpenAI API returned an error status: {e.status_code} - {e.response}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during OpenAI call: {e}")

    def _build_system_prompt(self) -> str:
        """
        Abstract method: Returns the system prompt specific to the agent's role.
        Must be implemented by subclasses.
        """
        return "You are a helpful AI assistant."

    def _enhance_request_with_context(self, request: str, context: AgentContext) -> str:
        """
        Helper method to format the user request with relevant business context.
        """
        context_str = f"""
BUSINESS CONTEXT:
- Type: {context.business_type if context.business_type else 'Not provided'}
- Industry: {context.industry if context.industry else 'Not provided'}
- Target Audience: {context.target_audience if context.target_audience else 'Not provided'}
- Current Revenue: {context.current_revenue if context.current_revenue else 'Not provided'}
- Main Challenges: {', '.join(context.main_challenges) if context.main_challenges else 'N/A'}
- Goals: {', '.join(context.goals) if context.goals else 'N/A'}
"""
        if context.previous_results:
            context_str += f"\nPREVIOUS WORK / CONTEXT: {json.dumps(context.previous_results, indent=2)}\n"
        if context.user_preferences:
            context_str += f"\nUSER PREFERENCES: {json.dumps(context.user_preferences, indent=2)}\n"
        if context.business_evolution_history:
             context_str += f"\nBUSINESS HISTORY (Latest): {json.dumps(context.business_evolution_history[-1], indent=2)}\n"
        if context.relationship_memory:
             context_str += f"\nRELATIONSHIP MEMORY: {json.dumps(context.relationship_memory, indent=2)}\n"
        if context.semantic_tags:
            context_str += f"\nSEMANTIC TAGS: {', '.join(context.semantic_tags)}\n"

        context_str += f"\nUSER REQUEST: {request}"
        return context_str

    def _structure_output(self, response_content: str) -> Dict[str, Any]:
        """
        Helper method to structure the raw LLM response into a dictionary.
        """
        return {"response": response_content}

    def _calculate_confidence(self, output: Dict[str, Any]) -> float:
        """
        Calculates a confidence score for the agent's output.
        """
        return 0.85

    def _recommend_next_agents(self, output: Dict[str, Any]) -> List[str]:
        """
        Recommends next agents to involve in a multi-step workflow based on output.
        """
        return []

    def _extract_context_updates(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts key pieces of information from the agent's output to update the global context.
        """
        return {}

    def _update_metrics(self, execution_time: float, cost: float, success: bool, confidence: float):
        """Updates the agent's internal performance metrics."""
        self.performance_metrics["total_executions"] += 1
        self.performance_metrics["total_cost"] += cost
        if success:
            self.performance_metrics["successful_executions"] += 1

        total = self.performance_metrics["total_executions"]
        successful = self.performance_metrics["successful_executions"]
        self.performance_metrics["success_rate"] = successful / total if total > 0 else 1.0

        current_avg_time = self.performance_metrics["avg_execution_time"]
        self.performance_metrics["avg_execution_time"] = (
            (current_avg_time * (total - 1)) + execution_time
        ) / total if total > 0 else execution_time

        current_avg_conf = self.performance_metrics["avg_confidence_score"]
        if success and confidence is not None:
             self.performance_metrics["avg_confidence_score"] = (
                (current_avg_conf * (successful - 1)) + confidence
             ) / successful if successful > 0 else confidence
        elif total == 0:
            self.performance_metrics["avg_confidence_score"] = 0.0