# backend/services/ai_service.py

import os
import aiohttp
import json # Import json for parsing LLM responses
from typing import Dict, List, Any

# This service is for "standard" models, potentially open-source or OpenRouter.
# It also includes the logic for whether to use elite agents in the first place.
class OpenSourceAIService:
    """
    Handles interactions with open-source compatible AI models (e.g., via OpenRouter.ai)
    and provides foundational AI services like task detection and elite agent routing decision.
    """
    def __init__(self, config: Dict):
        self.openrouter_key = config.get('openrouter_api_key')
        self.session: aiohttp.ClientSession = None # Will be initialized on startup

    async def initialize(self):
        """Initializes the aiohttp client session."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def cleanup(self):
        """Closes the aiohttp client session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    async def _api_call(self, messages: List[Dict], model: str) -> Dict[str, Any]:
        """Internal helper for making API calls to OpenRouter or similar."""
        if not self.session:
            raise RuntimeError("aiohttp session not initialized. Call initialize() first.")
        
        # --- CRITICAL FIX: Removed duplicate 'self' ---
        if not self.openrouter_key: # Corrected check for key
            raise ValueError("OPENROUTER_API_KEY is not configured for OpenSourceAIService.")

        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        try:
            async with self.session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            raise RuntimeError(f"API call failed for model {model} via OpenRouter: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during API call: {e}")

    async def chat_completion(self, messages: List[Dict], model: str) -> Dict[str, Any]:
        """
        Performs a chat completion using the specified model.
        Returns the response content and estimated cost.
        """
        data = await self._api_call(messages, model)
        
        total_tokens = data.get('usage', {}).get('total_tokens', 0)
        estimated_cost = total_tokens * 0.000001
        
        return {
            'response': data['choices'][0]['message']['content'],
            'cost': estimated_cost,
            'model': model,
            'provider': 'OpenRouter'
        }

    async def detect_task_type(self, user_prompt: str, classifier_model: str, valid_types: List[str]) -> str:
        """
        Detects the task type from a user prompt using a classifier model.
        """
        categories = ", ".join(f'"{t}"' for t in valid_types)
        system_prompt = f"Classify the user's message into one of these categories: {categories}. Respond with ONLY the category name in quotes."
        
        data = await self._api_call([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], classifier_model)
        
        detected_type = data['choices'][0]['message']['content'].strip().replace('"', '')
        return detected_type if detected_type in valid_types else "simple_qa"

    async def should_use_elite_agents(self, user_prompt: str, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Determines if a request should be routed to elite agents based on complexity.
        """
        system_prompt = """
You are a request complexity analyzer. Determine if this request requires elite AI agents (multi-step, specialized expertise, high-stakes) or can be handled by standard, general-purpose models.

ELITE AGENT INDICATORS:
- Business strategy, growth planning, scaling
- Marketing campaigns, conversion optimization, sales funnels
- Technical architecture, code optimization, system design, complex development tasks
- Deep data analysis, predictive modeling, complex insights
- Requests requiring coordination across multiple business functions or deep domain expertise
- Tasks with significant business impact or requiring highly tailored, comprehensive output

STANDARD MODEL INDICATORS:
- Simple questions and answers
- Basic information requests
- General creative writing (e.g., a simple poem, short story)
- Summarization of short texts
- General conversation or brainstorming
- Basic code snippets (not full architectural designs)

Respond with JSON. Do not include any other text or markdown.
{
  "use_elite_agents": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "why this decision was made",
  "suggested_workflow_type": "single" | "sequential" | "parallel" | "hybrid"
}
"""
        classifier_model_for_elite_decision = self.config.get('classifier_model', "openai/gpt-4o-mini")

        data = await self._api_call([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], classifier_model_for_elite_decision)

        try:
            result = json.loads(data['choices'][0]['message']['content'])
            if not isinstance(result, dict) or "use_elite_agents" not in result or "confidence" not in result:
                raise ValueError("LLM returned malformed JSON for elite agent decision.")
            
            result["use_elite_agents"] = result.get("use_elite_agents", False) and result.get("confidence", 0.0) >= threshold
            
            return result
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from should_use_elite_agents: {e} - Raw: {data['choices'][0]['message']['content']}")
            return {
                "use_elite_agents": False,
                "confidence": 0.2,
                "reasoning": "LLM response malformed, defaulting to standard routing.",
                "suggested_workflow_type": "single"
            }
        except Exception as e:
            print(f"Unexpected error in should_use_elite_agents: {e}")
            return {
                "use_elite_agents": False,
                "confidence": 0.1,
                "reasoning": "Unexpected error during elite agent decision, defaulting to standard.",
                "suggested_workflow_type": "single"
            }