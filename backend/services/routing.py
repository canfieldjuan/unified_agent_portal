# backend/services/routing.py

from typing import Dict, List
from collections import defaultdict
import structlog

logger = structlog.get_logger()

class SimpleIntelligentRouter:
    """
    A simple router that directs chat requests to appropriate models
    based on task type and user tier.
    """
    def __init__(self, config: Dict):
        self.model_tiers = config.get('model_tiers', {})
        self.task_tier_map = config.get('task_tier_map', {})
        self.model_providers = config.get('model_providers', {})
        self.round_robin_counter = defaultdict(int)

        if not self.model_tiers:
            logger.warning("No model tiers configured. Defaulting to a basic tier.")
            self.model_tiers = {"economy": ["openai/gpt-4o-mini"], "standard": ["openai/gpt-4o"]}
        if not self.task_tier_map:
            logger.warning("No task tier map configured. Defaulting all tasks to economy.")
            self.task_tier_map = {"default": "economy"}
        if not self.model_providers:
            logger.warning("No model providers configured.")
            self.model_providers = {"openai/gpt-4o-mini": "OpenAI", "openai/gpt-4o": "OpenAI"}

    def route_simple(self, task_type: str, user_tier: str) -> Dict[str, str]:
        """
        Routes a request to a specific model based on task type and user tier.
        Applies a basic round-robin within a tier's models.
        """
        tier_name = self.task_tier_map.get(task_type, 'economy')

        if user_tier == 'pro' and tier_name == 'economy':
            tier_name = 'standard'
        
        models_in_tier = self.model_tiers.get(tier_name)
        
        if not models_in_tier:
            logger.error(f"No models found for tier '{tier_name}'. Falling back to economy tier.")
            models_in_tier = self.model_tiers.get('economy', ["openai/gpt-4o-mini"])

        model_index = self.round_robin_counter[tier_name] % len(models_in_tier)
        selected_model = models_in_tier[model_index]
        self.round_robin_counter[tier_name] += 1

        provider = self.model_providers.get(selected_model, 'unknown')

        logger.info(
            "Standard chat routing decision",
            task_type=task_type,
            user_tier=user_tier,
            routed_tier=tier_name,
            selected_model=selected_model,
            provider=provider
        )

        return {
            'model': selected_model,
            'provider': provider,
            'reasoning': f"Detected '{task_type}' task, routed to {tier_name.upper()} tier model: {selected_model}"
        }