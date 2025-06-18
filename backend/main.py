# backend/main.py

import asyncio
import os
import sys
import time
import json
import shutil
from datetime import datetime
from collections import defaultdict
from functools import wraps
from typing import Dict, List, Any, Callable, Optional, Union

# Add parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- Third-Party Imports ---
import aiohttp
import uvicorn
import yaml # Still needed for initial config loading, though ConfigManager wraps it
import structlog
import openai  # Fixed: Added missing import
from fastapi import (
    FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse # Import HTMLResponse
from pydantic import ValidationError # Import ValidationError for more specific error handling
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# --- Minimal Stubs for Missing Classes ---
class ConfigManager:
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = {}
        self.load_config()
    
    def load_config(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
            else:
                # Default config
                self.config = {
                    "api": {
                        "host": "0.0.0.0",
                        "port": 8000,
                        "cors_origins": ["*"]
                    },
                    "logging": {
                        "level": "INFO",
                        "format": "structured",
                        "file": None
                    },
                    "database_url": "sqlite:///elite_agents.db",
                    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
                    "elite_agents": {
                        "enabled": True,
                        "auto_routing_threshold": 0.7,
                        "fallback_to_standard": True
                    }
                }
        except Exception as e:
            print(f"Error loading config: {e}")
            self.config = {}
    
    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

class OpenSourceAIService:
    def __init__(self, config):
        self.config = config
        self.session = None
    
    async def initialize(self):
        self.session = aiohttp.ClientSession()
    
    async def cleanup(self):
        if self.session:
            await self.session.close()
    
    async def should_use_elite_agents(self, message: str, threshold: float = 0.7):
        # Simple heuristic - if message contains business terms, route to elite agents
        business_keywords = ['revenue', 'business', 'grow', 'scale', 'optimize', 'convert', 'customers', 'marketing']
        score = sum(1 for keyword in business_keywords if keyword.lower() in message.lower()) / len(business_keywords)
        
        return {
            "use_elite_agents": score >= threshold,
            "confidence": score,
            "reasoning": f"Business keyword score: {score:.2f}"
        }
    
    async def detect_task_type(self, message: str, model: str = None, valid_types: List[str] = None):
        # Simple task type detection
        if not valid_types:
            valid_types = ['general', 'coding', 'writing', 'analysis']
        
        message_lower = message.lower()
        if any(word in message_lower for word in ['code', 'program', 'function', 'debug']):
            return 'coding'
        elif any(word in message_lower for word in ['write', 'article', 'content', 'copy']):
            return 'writing'
        elif any(word in message_lower for word in ['analyze', 'data', 'report', 'metrics']):
            return 'analysis'
        else:
            return 'general'
    
    async def chat_completion(self, messages: List[Dict], model: str = None):
        # Placeholder for standard chat completion
        return {
            "response": "This is a placeholder response from the standard AI service.",
            "cost": 0.001,
            "model": model or "gpt-3.5-turbo"
        }

class SimpleIntelligentRouter:
    def __init__(self, config):
        self.config = config
    
    def route_simple(self, task_type: str, user_tier: str = "free"):
        return {
            "model": "gpt-3.5-turbo",
            "provider": "openai", 
            "reasoning": f"Routed {task_type} task to standard model for {user_tier} user"
        }

class AgentContext:
    def __init__(self, business_type: str = "", industry: str = "", target_audience: str = "",
                 current_revenue: str = "", main_challenges: List[str] = None, goals: List[str] = None):
        self.business_type = business_type
        self.industry = industry
        self.target_audience = target_audience
        self.current_revenue = current_revenue
        self.main_challenges = main_challenges or []
        self.goals = goals or []

class AgentResult:
    def __init__(self, success: bool = True, output: Dict = None, cost: float = 0.0, 
                 confidence_score: float = 0.5, tool_calls: List = None, context_updates: Dict = None):
        self.success = success
        self.output = output or {}
        self.cost = cost
        self.confidence_score = confidence_score
        self.tool_calls = tool_calls or []
        self.context_updates = context_updates or {}

class BaseAgent:
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.status = type('Status', (), {'value': 'active'})()
        self.capabilities = [type('Cap', (), {'value': 'general'})()]
        self.performance_metrics = {"total_executions": 0, "success_rate": 1.0}
    
    async def execute(self, message: str, context: AgentContext) -> AgentResult:
        return AgentResult(
            success=True,
            output={"response": f"Agent processed: {message}"},
            cost=0.01,
            confidence_score=0.8
        )

# Agent stubs
class GrowthAssassin(BaseAgent): pass
class ConversionPredator(BaseAgent): pass
class InterfaceDestroyer(BaseAgent): pass
class EmailRevenueEngine(BaseAgent): pass
class DataOracle(BaseAgent): pass
class CodeOverlord(BaseAgent): pass
class SystemDominator(BaseAgent): pass
class SystemNexus(BaseAgent): pass

class IntentClassifier:
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    async def analyze_intent(self, message: str, context: AgentContext):
        return {
            "intent_type": "general_inquiry",
            "primary_capability": "general",
            "recommended_agents": ["Growth Assassin"],
            "workflow_type": "single",
            "complexity_score": 5,
            "estimated_time": "minutes",
            "context_requirements": ["business_context"],
            "reasoning": "Default routing for demonstration"
        }

class AgentOrchestrator:
    def __init__(self, agents: Dict):
        self.agents = agents
    
    async def execute_workflow(self, message: str, routing_decision: Dict, context: AgentContext):
        # Simple orchestration - execute first recommended agent
        recommended_agents = routing_decision.get("recommended_agents", ["Growth Assassin"])
        results = {}
        total_cost = 0.0
        
        for agent_name in recommended_agents:
            if agent_name in self.agents:
                result = await self.agents[agent_name].execute(message, context)
                results[agent_name] = result
                total_cost += result.cost
            else:
                # Create a dummy result for missing agents
                results[agent_name] = AgentResult(
                    success=False,
                    output={"error": f"Agent {agent_name} not found"},
                    cost=0.0
                )
        
        return {
            "success": len([r for r in results.values() if r.success]) > 0,
            "results": results,
            "total_cost": total_cost,
            "total_execution_time": 1.0,
            "workflow_type": routing_decision.get("workflow_type", "single"),
            "routing_decision": routing_decision,
            "tool_calls": [],
            "context_updates": {}
        }
    
    def get_system_metrics(self):
        return {
            "total_agents": len(self.agents),
            "active_agents": len(self.agents),
            "total_executions": 0,
            "success_rate": 1.0
        }

def handle_errors(func):
    """Decorator for handling errors in API endpoints"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return wrapper

# --- Try to import real classes, fall back to stubs ---
try:
    from backend.models import (
        Base, ChatHistory, ChatRequest, ChatResponse,
        FunctionCallRequest, FunctionCallResponse, DevelopmentTask,
        EliteAgentRequest, EliteAgentResponse
    )
except ImportError:
    print("Warning: Could not import models. Using minimal stubs.")
    from pydantic import BaseModel
    from sqlalchemy.ext.declarative import declarative_base
    
    Base = declarative_base()
    
    class ChatHistory(Base):
        __tablename__ = 'chat_history'
        id = None  # Will be created properly when real models are available
    
    class ChatRequest(BaseModel):
        message: str
        user_id: str = "anonymous"
        task_type: str = "auto"
        user_tier: str = "free"
        business_context: Optional[Dict] = None
        force_system_nexus: bool = False
        nexus_tool_target: Optional[str] = None
    
    class ChatResponse(BaseModel):
        success: bool
        response: str
        model: str
        provider: str
        cost: float
        response_time: float
        reasoning: str = ""
        agents_used: List[str] = []
        workflow_type: str = ""
        confidence_score: float = 0.0
        tool_calls_made: List = []
    
    class EliteAgentRequest(BaseModel):
        message: str
        user_id: str
        business_type: str = ""
        industry: str = ""
        target_audience: str = ""
        current_revenue: str = ""
        main_challenges: List[str] = []
        goals: List[str] = []
        call_nexus_tool: Optional[str] = None
        nexus_tool_args: Optional[Dict] = None
    
    class EliteAgentResponse(BaseModel):
        success: bool
        routing_decision: Dict
        workflow_result: Dict
        execution_time: float
        agents_used: List[str]
        error: Optional[str] = None
        tool_calls_made: List = []
    
    class DevelopmentTask(BaseModel):
        task_description: str


# --- Setup logging ---
def configure_logging(log_level: str = "INFO", log_format: str = "structured", log_file: Optional[str] = None):
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    if log_format == "structured":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True
    )
    
    import logging
    logging.basicConfig(level=getattr(logging, log_level.upper()), handlers=[])
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(file_handler)
    
    logging.getLogger("uvicorn.access").handlers = [
        logging.FileHandler(log_file) if log_file else logging.StreamHandler()
    ] if log_file else []
    logging.getLogger("uvicorn.error").handlers = [
        logging.FileHandler(log_file) if log_file else logging.StreamHandler()
    ] if log_file else []
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

logger = structlog.get_logger()


# --- ENHANCED UNIFIED AI PORTAL APPLICATION ---
class EnhancedUnifiedAIPortal:
    """
    The main FastAPI application for the Elite Agent Command Center.
    Initializes all services, agents, and API routes.
    """
    def __init__(self, config_file: str = "config.yaml"):
        logger.info("Initializing Elite Agent Command Center portal...")
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.config

        # Configure logging early based on loaded config
        configure_logging(
            self.config.get("logging.level", "INFO"),
            self.config.get("logging.format", "structured"),
            self.config.get("logging.file", None)
        )
        logger.info("Logging configured.")

        self.app = FastAPI(
            title="Elite Agent Command Center",
            version="4.0.0",
            description="Your unified portal for orchestrating killer AI agents and external system integrations."
        )

        # Initialize core AI services
        self.ai_service = OpenSourceAIService(self.config)
        self.router = SimpleIntelligentRouter(self.config)

        # Initialize OpenAI client (required by all killer agents)
        openai_api_key = self.config.get("openai_api_key")
        if openai_api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        else:
            logger.warning("No OpenAI API key found. Using mock client.")
            self.openai_client = None

        # Initialize killer agents (ensure they receive the openai_client)
        self.agents = {
            "Growth Assassin": GrowthAssassin(self.openai_client),
            "Conversion Predator": ConversionPredator(self.openai_client),
            "Interface Destroyer": InterfaceDestroyer(self.openai_client),
            "Email Revenue Engine": EmailRevenueEngine(self.openai_client),
            "Data Oracle": DataOracle(self.openai_client),
            "Code Overlord": CodeOverlord(self.openai_client),
            "System Dominator": SystemDominator(self.openai_client),
            "System Nexus": SystemNexus(self.openai_client)
        }
        logger.info(f"Initialized {len(self.agents)} killer agents.")

        # Initialize routing and orchestration
        self.intent_classifier = IntentClassifier(self.openai_client)
        self.elite_agent_center = AgentOrchestrator(self.agents)
        logger.info("Elite Agent Orchestrator initialized.")

        # Database setup
        db_url = self.config.get('database_url', 'sqlite:///elite_agents.db')
        self.db_engine = create_engine(db_url)
        # Only create tables if Base has proper metadata
        try:
            Base.metadata.create_all(self.db_engine)
        except Exception as e:
            logger.warning(f"Could not create database tables: {e}")
        self.DbSession = sessionmaker(autocommit=False, autoflush=False, bind=self.db_engine)
        logger.info(f"Database initialized: {db_url}")

        self.setup_app()
        logger.info("Elite Agent Command Center portal initialization complete.")

    def setup_app(self):
        """Configures FastAPI middleware, static files, and event handlers."""
        # Create directories
        for directory in ["uploads", "output_files"]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created {directory} directory")

        # Frontend directory detection
        self.frontend_dir = None
        for dir_name in ["frontend", "Frontend"]:
            if os.path.exists(dir_name):
                self.frontend_dir = dir_name
                break
        
        if self.frontend_dir:
            self.app.mount("/static", StaticFiles(directory=self.frontend_dir), name="static")
            logger.info(f"Frontend static files mounted from: {self.frontend_dir}")
        else:
            logger.warning("No frontend directory found. Frontend UI may not be available.")

        # CORS Middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get("api.cors_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )

        # FastAPI event handlers
        @self.app.on_event("startup")
        async def startup_event():
            logger.info("FastAPI startup event triggered.")
            await self.ai_service.initialize()
            logger.info("OpenSourceAIService aiohttp session initialized.")

        @self.app.on_event("shutdown")
        async def shutdown_event():
            logger.info("FastAPI shutdown event triggered.")
            await self.ai_service.cleanup()
            logger.info("OpenSourceAIService aiohttp session closed.")

        self.setup_routes()
        logger.info("API routes setup complete.")

    def setup_routes(self):
        """Defines all the API endpoints for the application."""

        @self.app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def root():
            """Serves the main frontend HTML page or a basic status page."""
            return HTMLResponse(content=f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Elite Agent Command Center</title>
                    <style>
                        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1000px; margin: 50px auto; padding: 20px; background: #0f1419; color: #e6e6e6; }}
                        .container {{ background: #1a1f2e; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }}
                        h1 {{ color: #4f46e5; margin-bottom: 10px; font-size: 2.5rem; }}
                        .status {{ background: #16a34a; padding: 15px; border-radius: 10px; color: white; margin: 20px 0; font-weight: bold; }}
                        .elite-status {{ background: #7c3aed; padding: 15px; border-radius: 10px; color: white; margin: 20px 0; }}
                        .links {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 30px 0; }}
                        .links a {{ display: block; padding: 15px; background: #4f46e5; color: white; text-decoration: none; border-radius: 10px; text-align: center; transition: background 0.2s; }}
                        .links a:hover {{ background: #6366f1; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>🤖 Elite Agent Command Center</h1>
                        
                        <div class="status">
                            ✅ Backend Online - Elite Intelligence Ready (Demo Mode)
                        </div>
                        
                        <div class="elite-status">
                            🔥 Elite Agents: Active (Demo) |
                            Killer Agents: {len(self.elite_agent_center.agents) if self.elite_agent_center else 0} |
                            Auto-Routing: Enabled
                        </div>
                        
                        <div class="links">
                            <a href="/docs">API Documentation</a>
                            <a href="/elite/agents/status">Agent Status</a>
                            <a href="/elite/metrics">System Metrics</a>
                            <a href="/health">Health Check</a>
                        </div>
                        
                        <h3>🚀 Quick Test:</h3>
                        <pre>POST /elite/chat
{{
  "message": "Help me grow my business from $50K to $500K MRR",
  "user_id": "test-user",
  "business_type": "SaaS",
  "industry": "Marketing Technology"
}}</pre>
                        
                        <div style="background: #f59e0b; padding: 15px; border-radius: 10px; color: black; margin: 20px 0;">
                            ⚠️ Running in Demo Mode - Some modules are using placeholder implementations
                        </div>
                    </div>
                </body>
                </html>
            """)

        @self.app.get("/health")
        @handle_errors
        async def health_check():
            """Returns the health status of the application and agent system."""
            return {
                "status": "healthy",
                "message": "Elite Agent Command Center is online (Demo Mode)",
                "elite_agents_available": self.elite_agent_center is not None and bool(self.elite_agent_center.agents),
                "agents_count": len(self.elite_agent_center.agents) if self.elite_agent_center else 0,
                "database_connection": "OK" if self.db_engine else "Failed",
                "config_loaded": bool(self.config),
                "demo_mode": True
            }

        @self.app.post("/elite/chat", response_model=EliteAgentResponse, tags=["Elite Agents"])
        @handle_errors
        async def elite_chat_endpoint(request: EliteAgentRequest, background_tasks: BackgroundTasks):
            """Processes a request through the Elite Agent Command Center."""
            try:
                if not self.elite_agent_center:
                    raise HTTPException(status_code=503, detail="Elite Agent Command Center not available.")

                start_time = time.time()
                
                # Create AgentContext from the request
                agent_context = AgentContext(
                    business_type=request.business_type,
                    industry=request.industry,
                    target_audience=request.target_audience,
                    current_revenue=request.current_revenue,
                    main_challenges=request.main_challenges,
                    goals=request.goals
                )
                
                # Analyze intent using the IntentClassifier
                routing_decision = await self.intent_classifier.analyze_intent(
                    request.message,
                    agent_context
                )
                
                # Execute workflow based on classification
                result_from_orchestrator = await self.elite_agent_center.execute_workflow(
                    request.message,
                    routing_decision,
                    agent_context
                )

                execution_time = time.time() - start_time

                if result_from_orchestrator["success"]:
                    # Combine agent outputs
                    combined_response_parts = []
                    total_confidence = 0.0
                    agents_contributing = 0

                    for agent_name, agent_result in result_from_orchestrator["results"].items():
                        if agent_result.success:
                            agent_output_text = agent_result.output.get('response') or json.dumps(agent_result.output, indent=2)
                            combined_response_parts.append(f"**{agent_name}:**\n{agent_output_text}")
                            total_confidence += agent_result.confidence_score
                            agents_contributing += 1
                        else:
                            combined_response_parts.append(f"**{agent_name} (Failed):**\nError: {agent_result.output.get('error', 'Unknown Error')}")

                    avg_confidence = total_confidence / agents_contributing if agents_contributing > 0 else 0.0
                    combined_response = "\n\n".join(combined_response_parts)

                    elite_api_response = EliteAgentResponse(
                        success=True,
                        routing_decision=routing_decision,
                        workflow_result=result_from_orchestrator,
                        execution_time=execution_time,
                        agents_used=result_from_orchestrator["routing_decision"].get("recommended_agents", []),
                        tool_calls_made=result_from_orchestrator["tool_calls"]
                    )

                    return elite_api_response
                else:
                    raise HTTPException(status_code=500, detail=result_from_orchestrator.get("error", "Elite agent orchestration failed."))

            except ValidationError as e:
                logger.error("Pydantic validation error for EliteAgentRequest", exc_info=True)
                raise HTTPException(status_code=422, detail=f"Invalid request payload: {str(e)}")
            except Exception as e:
                logger.error("Unhandled error in elite_chat_endpoint", exc_info=True)
                raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

        @self.app.get("/elite/agents/status", tags=["Elite Agents"])
        @handle_errors
        async def get_elite_agent_status():
            """Gets the operational status and basic performance metrics of all elite agents."""
            if not self.elite_agent_center or not self.elite_agent_center.agents:
                return {"error": "Elite agents not initialized or available."}
            
            status_data = {}
            for agent_name, agent_instance in self.elite_agent_center.agents.items():
                status_data[agent_name] = {
                    "status": agent_instance.status.value,
                    "capabilities": [cap.value for cap in agent_instance.capabilities],
                    "performance": agent_instance.performance_metrics
                }
            return status_data

        @self.app.get("/elite/metrics", tags=["Elite Agents"])
        @handle_errors
        async def get_elite_metrics():
            """Gets overall system performance metrics for the Elite Agent Command Center."""
            if not self.elite_agent_center:
                return {"error": "Elite Agent Command Center not initialized."}
            
            return self.elite_agent_center.get_system_metrics()

        @self.app.post("/chat", response_model=ChatResponse, tags=["Standard Chat"])
        @handle_errors
        async def enhanced_chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
            """Enhanced chat endpoint with intelligent routing."""
            start_time = time.time()

            try:
                # Simple routing logic
                final_task_type = await self.ai_service.detect_task_type(request.message)
                routing_decision_standard = self.router.route_simple(final_task_type, request.user_tier)
                result_standard_chat = await self.ai_service.chat_completion([
                    {"role": "user", "content": request.message}
                ], routing_decision_standard['model'])

                execution_time = time.time() - start_time
                chat_response_standard = ChatResponse(
                    success=True,
                    response=result_standard_chat['response'],
                    model=routing_decision_standard['model'],
                    provider=routing_decision_standard['provider'],
                    cost=result_standard_chat['cost'],
                    response_time=execution_time,
                    reasoning=routing_decision_standard['reasoning']
                )
                return chat_response_standard

            except Exception as e:
                logger.error("Error in enhanced_chat_endpoint", exc_info=True)
                raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    def _save_chat_history(self, user_id: str, message: str, response: ChatResponse, context_updates: Dict[str, Any]):
        """Background task to save chat history and context updates."""
        try:
            # Simplified history saving
            logger.info("Chat history would be saved", user_id=user_id, model=response.model)
        except Exception as e:
            logger.error("Failed to save chat history", user_id=user_id, error=str(e), exc_info=True)

    def run(self):
        """Runs the FastAPI server using Uvicorn."""
        host = self.config.get("api.host", "0.0.0.0")
        port = self.config.get("api.port", 8000)
        
        logger.info(f"🚀 Starting Elite Agent Command Center on http://{host}:{port}")
        logger.info(f"🔥 Elite Agents: Active (Demo Mode)")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level=self.config.get("logging.level", "info").lower(),
            access_log=True
        )

# Global portal instance (used by setup.sh and main entry point)
portal = EnhancedUnifiedAIPortal()
app = portal.app # Expose the FastAPI app instance for Uvicorn when run directly

if __name__ == "__main__":
    try:
        portal.run()
    except Exception as e:
        logger.critical(f"Failed to start Elite Agent Command Center: {e}", exc_info=True)