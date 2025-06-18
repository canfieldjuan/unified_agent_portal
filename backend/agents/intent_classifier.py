# backend/agents/intent_classifier.py

import json
import openai
from typing import Dict, List, Any

from backend.agents.base_agent import AgentContext
import structlog

logger = structlog.get_logger()

class IntentClassifier:
    """
    Classifies user intent and recommends the optimal agent strategy (single agent,
    multi-agent workflow, or external tool interaction via System Nexus).
    """

    def __init__(self, openai_client: Optional[openai.AsyncOpenAI]): # Added Optional for client
        self.client = openai_client

    async def analyze_intent(self, user_request: str, context: AgentContext) -> Dict[str, Any]:
        """
        Analyzes user intent and recommends an agent routing strategy,
        including routing to the System Nexus if an external tool is needed.
        """
        if self.client is None: # Added check for client initialization
            logger.error("OpenAI client not initialized for IntentClassifier. Cannot analyze intent with LLM.")
            return self._fallback_routing(user_request, "OpenAI client not available")

        system_prompt = self._build_system_prompt()

        enhanced_request = f"""
USER REQUEST: {user_request}

BUSINESS CONTEXT:
- Type: {context.business_type if context.business_type else 'Not provided'}
- Industry: {context.industry if context.industry else 'Not provided'}
- Challenges: {', '.join(context.main_challenges) if context.main_challenges else 'Not provided'}
- Goals: {', '.join(context.goals) if context.goals else 'Not provided'}
- Previous Results: {json.dumps(context.previous_results) if context.previous_results else 'None'}

Analyze this request and determine optimal agent routing based on available agents and tools.
"""
        try:
            response = await self.client.chat.completions.acreate(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": enhanced_request}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            llm_output = response.choices[0].message.content
            
            try:
                result = json.loads(llm_output)
                if not all(k in result for k in ["intent_type", "primary_capability", "recommended_agents", "workflow_type"]):
                    raise ValueError("Missing required keys in LLM's JSON response.")
                logger.info("Intent classified successfully", routing_decision=result)
                return result
            except json.JSONDecodeError as e:
                logger.warning("LLM output was not valid JSON, attempting fallback routing.", error=str(e), llm_output=llm_output)
                return self._fallback_routing(user_request, llm_output)
            except ValueError as e:
                logger.warning(f"LLM output JSON missing keys, attempting fallback routing. {e}", llm_output=llm_output)
                return self._fallback_routing(user_request, llm_output)

        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API connection error during intent classification: {e}")
            return self._fallback_routing(user_request, "API Connection Error")
        except openai.RateLimitError as e:
            logger.error(f"OpenAI API rate limit exceeded during intent classification: {e}")
            return self._fallback_routing(user_request, "API Rate Limit Exceeded")
        except openai.APIStatusError as e:
            logger.error(f"OpenAI API status error during intent classification: {e.status_code} - {e.response}")
            return self._fallback_routing(user_request, f"API Status Error: {e.status_code}")
        except Exception as e:
            logger.error("Unexpected error during intent classification", exc_info=True, error=str(e))
            return self._fallback_routing(user_request, "Unexpected Classification Error")

    def _build_system_prompt(self) -> str:
        """
        Defines the system prompt for the intent classifier, listing all available
        agents and the new System Nexus's role for external tools.
        """
        return """
You are an elite intent classification system for coordinating killer AI agents.
Your task is to analyze user requests and determine the optimal routing strategy.

AVAILABLE KILLER AGENTS:
1. Growth Assassin - Business strategy, growth constraints, scaling plans.
2. Conversion Predator - Marketing copy, sales funnels, conversion optimization.
3. Interface Destroyer - UI/UX design, landing pages, user experience.
4. Email Revenue Engine - Email marketing, automation, revenue sequences.
5. Data Oracle - Analytics, insights, predictive modeling, customer intelligence.
6. Code Overlord - Code optimization, architecture, performance improvements, code generation.
7. System Dominator - Infrastructure, scalability, system architecture.
8. System Nexus - **Crucial for external integrations**. Use this agent when the request explicitly involves interacting with or triggering actions in systems like CRM (Salesforce, HubSpot), project management (Jira, Asana), cloud services (Google Cloud, AWS), marketing platforms (Mailchimp, HubSpot Marketing), or fetching/writing data to external databases/APIs (e.g., "update Salesforce lead," "get latest data from Stripe," "trigger n8n workflow," "run a CodeGPT task").

TASK: Analyze the user's request, considering the provided business context, and output a JSON response with the best routing.

JSON SCHEMA:
{
  "intent_type": "single_agent" | "multi_agent" | "sequential_workflow" | "parallel_workflow" | "external_tool_call",
  "primary_capability": "growth_strategy" | "conversion_optimization" | "design_systems" | "email_marketing" | "data_analysis" | "code_optimization" | "system_architecture" | "external_integration" | "unknown",
  "recommended_agents": ["Agent Name 1", "Agent Name 2"],
  "workflow_type": "sequential" | "parallel" | "hybrid" | "single",
  "complexity_score": 1-10,
  "estimated_time": "minutes" | "hours" | "days",
  "context_requirements": ["business_info", "technical_specs", "user_data", "crm_access", "code_repo_access", "etc"],
  "reasoning": "Why this routing decision was made"
}

RESPOND ONLY WITH VALID JSON.
"""

    def _fallback_routing(self, user_prompt: str, error_reason: str) -> Dict[str, Any]:
        """
        Provides a default routing decision if the primary classification fails.
        """
        logger.warning("Falling back to default routing for intent classification.", user_prompt=user_prompt, reason=error_reason)

        if any(keyword in user_prompt.lower() for keyword in ["growth", "scale", "mrr", "strategy", "business plan"]):
            primary_agent = "Growth Assassin"
            capability = "growth_strategy"
        elif any(keyword in user_prompt.lower() for keyword in ["copy", "marketing", "ad", "sales funnel", "conversion"]):
            primary_agent = "Conversion Predator"
            capability = "conversion_optimization"
        elif any(keyword in user_prompt.lower() for keyword in ["ui", "ux", "design", "interface", "landing page"]):
            primary_agent = "Interface Destroyer"
            capability = "design_systems"
        elif any(keyword in user_prompt.lower() for keyword in ["email", "campaign", "newsletter", "sequence"]):
            primary_agent = "Email Revenue Engine"
            capability = "email_marketing"
        elif any(keyword in user_prompt.lower() for keyword in ["data", "analyze", "metrics", "insights", "predictive"]):
            primary_agent = "Data Oracle"
            capability = "data_analysis"
        elif any(keyword in user_prompt.lower() for keyword in ["code", "optimize", "bug", "refactor", "api"]):
            primary_agent = "Code Overlord"
            capability = "code_optimization"
        elif any(keyword in user_prompt.lower() for keyword in ["system", "architecture", "scale", "infrastructure", "deploy"]):
            primary_agent = "System Dominator"
            capability = "system_architecture"
        elif any(keyword in user_prompt.lower() for keyword in ["connect to", "update", "pull from", "push to", "trigger", "login", "authenticate", "integrate"]):
            primary_agent = "System Nexus"
            capability = "external_integration"
        else:
            primary_agent = "Growth Assassin"
            capability = "growth_strategy"

        return {
            "intent_type": "single_agent",
            "primary_capability": capability,
            "recommended_agents": [primary_agent],
            "workflow_type": "single",
            "complexity_score": 3,
            "estimated_time": "minutes",
            "context_requirements": ["business_info"],
            "reasoning": f"Fallback routing due to classification error or unparsable LLM response. Defaulted based on keywords and assigned {primary_agent}."
        }