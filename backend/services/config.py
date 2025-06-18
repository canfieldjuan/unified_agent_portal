# backend/services/config.py

import os
import yaml
import structlog

logger = structlog.get_logger()

class ConfigManager:
    """Manages application configuration loaded from a YAML file and environment variables."""

    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = {}
        self.load_config()

    def load_config(self):
        """Loads configuration from the YAML file, then overrides with environment variables."""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Config file not found, using default configuration.", config_file=self.config_file)
            self.config = self._get_default_config()
        except yaml.YAMLError as e:
            logger.error("Error parsing config.yaml, using default configuration.", error=str(e))
            self.config = self._get_default_config()

        self._override_with_env_vars()

        logger.info(
            "Configuration loaded.",
            has_openai_key=bool(self.get("openai_api_key")),
            has_openrouter_key=bool(self.get("openrouter_api_key")),
            has_gemini_key=bool(self.get("gemini_api_key")),
            database_url_configured=self.get("database_url", "not_set") != "sqlite:///ai_portal.db"
        )

    def _override_with_env_vars(self):
        """Overrides config values with corresponding environment variables."""
        env_map = {
            "OPENAI_API_KEY": "openai_api_key",
            "OPENROUTER_API_KEY": "openrouter_api_key",
            "GEMINI_API_KEY": "gemini_api_key",
            "DATABASE_URL": "database_url",
            "API_HOST": "api.host",
            "API_PORT": "api.port",
            "ENVIRONMENT": "environment",
            "LOG_LEVEL": "logging.level"
        }

        for env_var, config_path in env_map.items():
            if os.environ.get(env_var):
                keys = config_path.split('.')
                current_level = self.config
                for i, key in enumerate(keys):
                    if i == len(keys) - 1:
                        if key == 'port':
                            try:
                                current_level[key] = int(os.environ[env_var])
                            except ValueError:
                                logger.error(f"Invalid value for {env_var}. Port must be an integer.")
                        else:
                            current_level[key] = os.environ[env_var]
                    else:
                        current_level = current_level.setdefault(key, {})

    def _get_default_config(self) -> dict:
        """Returns a dictionary of default configuration settings."""
        return {
            "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
            "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
            "classifier_model": "openai/gpt-4o-mini",
            "database_url": "sqlite:///ai_portal.db",
            "valid_task_types": ["simple_qa", "code_generation", "creative_writing", "business_strategy"],
            "model_tiers": {
                "economy": ["openai/gpt-4o-mini"],
                "standard": ["openai/gpt-4o"],
                "premium": ["anthropic/claude-3.5-sonnet"]
            },
            "task_tier_map": {
                "simple_qa": "economy",
                "code_generation": "standard",
                "creative_writing": "premium",
                "business_strategy": "premium"
            },
            "model_providers": {
                "openai/gpt-4o-mini": "OpenAI",
                "openai/gpt-4o": "OpenAI",
                "anthropic/claude-3.5-sonnet": "Anthropic"
            },
            "elite_agents": {
                "enabled": True,
                "auto_routing_threshold": 0.7,
                "fallback_to_standard": True,
                "max_concurrent_agents": 3,
                "timeout_seconds": 300
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            },
            "logging": {
                "level": "INFO",
                "format": "structured",
                "file": "elite_agents.log"
            },
            "performance_thresholds": {
                "min_confidence_score": 0.7,
                "max_response_time": 120,
                "max_cost_per_request": 1.0
            }
        }

    def get(self, key: str, default=None):
        """Retrieves a configuration value using dot notation for nested keys (e.g., 'api.host')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value