# backend/main.py

import asyncio
import os
import time
import json
import shutil
from datetime import datetime
from collections import defaultdict
from functools import wraps
from typing import Dict, List, Any, Callable, Optional, Union

# --- Third-Party Imports ---
import aiohttp
import uvicorn
import yaml
import structlog
import openai
from fastapi import (
    FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# --- Local Module Imports (New Structure) ---
from backend.models import (
    Base, ChatHistory, ChatRequest, ChatResponse,
    FunctionCallRequest, FunctionCallResponse, DevelopmentTask,
    EliteAgentRequest, EliteAgentResponse
)
from backend.services.config import ConfigManager
from backend.services.ai_service import OpenSourceAIService
from backend.services.routing import SimpleIntelligentRouter
from backend.services.orchestration import AgentOrchestrator
from backend.services.utils import handle_errors

from backend.agents.base_agent import AgentContext
from backend.agents.killer_agents import (
    GrowthAssassin, ConversionPredator, InterfaceDestroyer,
    EmailRevenueEngine, DataOracle, CodeOverlord, SystemDominator
)
from backend.agents.system_nexus import SystemNexus
from backend.agents.intent_classifier import IntentClassifier


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

        self.ai_service = OpenSourceAIService(self.config)
        self.router = SimpleIntelligentRouter(self.config)

        # Initialize OpenAI client (required by all killer agents)
        openai_api_key = self.config.get("openai_api_key")
        if openai_api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        else:
            logger.warning("No OpenAI API key found. Elite agents requiring OpenAI will not function.")
            self.openai_client = None # Set to None if no API key, agents need to handle this

        # Initialize killer agents (ensure they receive the openai_client)
        # Agents requiring OpenAI client will need to handle the None case if client is None
        self.agents = {
            "Growth Assassin": GrowthAssassin(self.openai_client),
            "Conversion Predator": ConversionPredator(self.openai_client),
            "Interface Destroyer": InterfaceDestroyer(self.openai_client),
            "Email Revenue Engine": EmailRevenueEngine(self.openai_client),
            "Data Oracle": DataOracle(self.openai_client),
            "Code Overlord": CodeOverlord(self.openai_client),
            "System Dominator": SystemDominator(self.openai_client),
            "System Nexus": SystemNexus(self.openai_client) # Instantiate the new System Nexus
        }
        logger.info(f"Initialized {len(self.agents)} killer agents.")

        self.intent_classifier = IntentClassifier(self.openai_client)
        self.elite_agent_center = AgentOrchestrator(self.agents)
        logger.info("Elite Agent Orchestrator initialized.")

        # Database setup
        db_url = self.config.get('database_url', 'sqlite:///elite_agents.db')
        self.db_engine = create_engine(db_url)
        try:
            Base.metadata.create_all(self.db_engine)
            logger.info(f"Database tables created or already exist.")
        except Exception as e:
            logger.error(f"Failed to create database tables. This might indicate a problem with models.py or database connection.", error=str(e), exc_info=True)
            # You might want to halt execution or run in a degraded mode if DB is critical
        self.DbSession = sessionmaker(autocommit=False, autoflush=False, bind=self.db_engine)
        logger.info(f"Database initialized: {db_url}")

        self.setup_app()
        logger.info("Elite Agent Command Center portal initialization complete.")

    def setup_app(self):
        """Configures FastAPI middleware, static files, and event handlers."""
        for directory in ["uploads", "output_files", "logs"]: # Ensure logs dir is also created
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created {directory} directory")
        
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

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get("api.cors_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )

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

        # ============================================================================
        # HEALTH CHECK & ROOT ENDPOINTS
        # ============================================================================

        @self.app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def root():
            """Serves the main frontend HTML page or a basic status page."""
            if self.frontend_dir:
                try:
                    with open(os.path.join(self.frontend_dir, "index.html"), "r", encoding="utf-8") as f:
                        content = f.read()
                        content = content.replace('href="/frontend/style.css"', 'href="/static/enhanced-style.css"')
                        content = content.replace('src="/frontend/script.js"', 'src="/static/enhanced-script.js"')
                        return HTMLResponse(content=content)
                except FileNotFoundError:
                    logger.error(f"index.html not found in {self.frontend_dir}")
                    pass
            
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
                        .agent-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
                        .agent-card {{ background: #2d3748; padding: 20px; border-radius: 10px; border-left: 4px solid #4f46e5; }}
                        .agent-name {{ font-weight: bold; color: #4f46e5; margin-bottom: 10px; }}
                        .agent-desc {{ font-size: 0.9rem; color: #a0a0a0; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>🤖 Elite Agent Command Center</h1>
                        
                        <div class="status">
                            ✅ Backend Online - Elite Intelligence Ready
                        </div>
                        
                        <div class="elite-status">
                            🔥 Elite Agents: {'Active' if self.elite_agent_center and len(self.elite_agent_center.agents) > 0 else 'Offline'} |
                            Killer Agents: {len(self.elite_agent_center.agents) if self.elite_agent_center else 0} |
                            Auto-Routing: {'Enabled' if self.config.get("elite_agents.enabled") else 'Disabled'}
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
                "message": "Elite Agent Command Center is online",
                "elite_agents_available": self.elite_agent_center is not None and bool(self.elite_agent_center.agents),
                "agents_count": len(self.elite_agent_center.agents) if self.elite_agent_center else 0,
                "database_connection": "OK" if self.db_engine else "Failed",
                "config_loaded": bool(self.config)
            }
        
        @self.app.get("/style.css", include_in_schema=False)
        @self.app.get("/static/style.css", include_in_schema=False)
        async def serve_css():
            if self.frontend_dir:
                try:
                    with open(os.path.join(self.frontend_dir, "enhanced-style.css"), "r", encoding="utf-8") as f:
                        return Response(content=f.read(), media_type="text/css")
                except FileNotFoundError:
                    pass
            raise HTTPException(status_code=404, detail="CSS file not found")

        @self.app.get("/script.js", include_in_schema=False)
        @self.app.get("/static/script.js", include_in_schema=False)
        async def serve_js():
            if self.frontend_dir:
                try:
                    with open(os.path.join(self.frontend_dir, "enhanced-script.js"), "r", encoding="utf-8") as f:
                        return Response(content=f.read(), media_type="application/javascript")
                except FileNotFoundError:
                    pass
            raise HTTPException(status_code=404, detail="JS file not found")


        # ============================================================================
        # ELITE AGENT ENDPOINTS
        # ============================================================================

        @self.app.post("/elite/chat", response_model=EliteAgentResponse, tags=["Elite Agents"])
        @handle_errors
        async def elite_chat_endpoint(request: EliteAgentRequest, background_tasks: BackgroundTasks):
            """
            Processes a request through the Elite Agent Command Center.
            The system will analyze the intent and orchestrate relevant agents (including System Nexus).
            """
            # The try...except block needs to be inside the function body
            try: # <--- THIS IS THE CRITICAL 'try:' FOR /elite/chat
                if not self.elite_agent_center:
                    raise HTTPException(status_code=503, detail="Elite Agent Command Center not available.")

                start_time = time.time()
                
                # Check for direct System Nexus tool call requested from EliteAgentRequest
                if request.call_nexus_tool and request.nexus_tool_args:
                    logger.info("Direct System Nexus tool call requested via /elite/chat endpoint.", tool=request.call_nexus_tool)
                    routing_decision = {
                        "intent_type": "external_tool_call",
                        "primary_capability": "external_integration",
                        "recommended_agents": ["System Nexus"],
                        "workflow_type": "single",
                        "complexity_score": 5,
                        "estimated_time": "minutes",
                        "context_requirements": ["tool_specific"],
                        "reasoning": f"Direct tool call requested for {request.call_nexus_tool}."
                    }
                    nexus_message = (
                        f"Perform the following action using the specified tool: "
                        f"Tool: {request.call_nexus_tool}, Arguments: {json.dumps(request.nexus_tool_args)}. "
                        f"Reason: API direct call."
                        f"\nOriginal message context: {request.message}"
                    )
                    
                    agent_context = AgentContext(
                        business_type=request.business_type,
                        industry=request.industry,
                        target_audience=request.target_audience,
                        current_revenue=request.current_revenue,
                        main_challenges=request.main_challenges,
                        goals=request.goals
                    )

                    result_from_orchestrator = await self.elite_agent_center.execute_workflow(
                        nexus_message,
                        routing_decision,
                        agent_context
                    )

                else:
                    agent_context = AgentContext(
                        business_type=request.business_type,
                        industry=request.industry,
                        target_audience=request.target_audience,
                        current_revenue=request.current_revenue,
                        main_challenges=request.main_challenges,
                        goals=request.goals
                    )
                    routing_decision = await self.intent_classifier.analyze_intent(
                        request.message,
                        agent_context
                    )
                    
                    result_from_orchestrator = await self.elite_agent_center.execute_workflow(
                        request.message,
                        routing_decision,
                        agent_context
                    )

                execution_time = time.time() - start_time

                if result_from_orchestrator["success"]:
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

                    chat_response_for_db = ChatResponse(
                        success=True,
                        response=combined_response,
                        model="Elite Agent Team",
                        provider="Elite Command Center",
                        cost=result_from_orchestrator["total_cost"],
                        response_time=execution_time,
                        agents_used=result_from_orchestrator["routing_decision"].get("recommended_agents", []),
                        workflow_type=result_from_orchestrator["workflow_type"],
                        confidence_score=avg_confidence,
                        tool_calls_made=result_from_orchestrator["tool_calls"]
                    )
                    
                    elite_api_response = EliteAgentResponse(
                        success=True,
                        routing_decision=routing_decision,
                        workflow_result=result_from_orchestrator,
                        execution_time=execution_time,
                        agents_used=result_from_orchestrator["routing_decision"].get("recommended_agents", []),
                        tool_calls_made=result_from_orchestrator["tool_calls"]
                    )

                    background_tasks.add_task(
                        self._save_chat_history,
                        request.user_id,
                        request.message,
                        chat_response_for_db,
                        result_from_orchestrator["context_updates"]
                    )
                    return elite_api_response
                else:
                    logger.error("Elite Agent orchestration failed for request", user_id=request.user_id, error=result_from_orchestrator.get("error", "Unknown orchestration error"))
                    raise HTTPException(status_code=500, detail=result_from_orchestrator.get("error", "Elite agent orchestration failed."))

            except ValidationError as e:
                logger.error("Pydantic validation error for EliteAgentRequest", exc_info=True, errors=e.errors())
                raise HTTPException(status_code=422, detail=f"Invalid request payload: {e.errors()}")
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


        @self.app.post("/elite/agents/{agent_name}", tags=["Elite Agents"], response_model=EliteAgentResponse)
        @handle_errors
        async def call_specific_elite_agent(
            agent_name: str,
            request: EliteAgentRequest,
            background_tasks: BackgroundTasks
        ):
            """
            Directly calls a specific elite agent by name.
            Useful for testing or specific integrations that know which agent to target.
            """
            # The try...except block needs to be inside the function body
            try: # <--- THIS IS THE CRITICAL 'try:' FOR /elite/agents/{agent_name}
                if not self.elite_agent_center:
                    raise HTTPException(status_code=503, detail="Elite Agent Command Center not available.")
                
                if agent_name not in self.elite_agent_center.agents:
                    raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
                
                start_time = time.time()
                
                agent_context = AgentContext(
                    business_type=request.business_type,
                    industry=request.industry,
                    target_audience=request.target_audience,
                    current_revenue=request.current_revenue,
                    main_challenges=request.main_challenges,
                    goals=request.goals
                )

                agent_instance = self.elite_agent_center.agents[agent_name]
                result = await agent_instance.execute(request.message, agent_context)
                
                execution_time = time.time() - start_time
                
                routing_decision_simulated = {
                    "intent_type": "single_agent",
                    "primary_capability": agent_name.replace(" ", "_").lower(),
                    "recommended_agents": [agent_name],
                    "workflow_type": "single",
                    "complexity_score": 5,
                    "estimated_time": "minutes",
                    "reasoning": f"Direct call to agent {agent_name}."
                }

                chat_response_for_db = ChatResponse(
                    success=result.success,
                    response=json.dumps(result.output) if isinstance(result.output, dict) else str(result.output),
                    model=agent_name,
                    provider="Elite Command Center",
                    cost=result.cost,
                    response_time=execution_time,
                    agents_used=[agent_name],
                    workflow_type="single",
                    confidence_score=result.confidence_score,
                    tool_calls_made=result.tool_calls
                )

                background_tasks.add_task(
                    self._save_chat_history,
                    request.user_id,
                    request.message,
                    chat_response_for_db,
                    result.context_updates
                )
                
                return EliteAgentResponse(
                    success=result.success,
                    routing_decision=routing_decision_simulated,
                    workflow_result={"results": {agent_name: result}, "total_cost": result.cost, "total_execution_time": execution_time, "context_updates": result.context_updates},
                    execution_time=execution_time,
                    agents_used=[agent_name],
                    error=result.output.get('error') if not result.success else None,
                    tool_calls_made=result.tool_calls
                )

            except ValidationError as e:
                logger.error("Pydantic validation error for EliteAgentRequest", exc_info=True, errors=e.errors())
                raise HTTPException(status_code=422, detail=f"Invalid request payload: {e.errors()}")
            except Exception as e:
                logger.error("Unhandled error in call_specific_elite_agent", exc_info=True)
                raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

        # ============================================================================
        # ENHANCED STANDARD CHAT ENDPOINT (now routes to Elite Agents if complex)
        # ============================================================================

        @self.app.post("/chat", response_model=ChatResponse, tags=["Standard Chat"])
        @handle_errors
        async def enhanced_chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
            """
            Enhanced chat endpoint with intelligent routing to Elite Agents or standard models.
            Automatically detects complexity and routes accordingly.
            """
            # The try...except block needs to be inside the function body
            try: # <--- THIS IS THE CRITICAL 'try:' FOR /chat
                start_time = time.time()

                user_agent_context = AgentContext(
                    business_type=request.business_context.get("business_type") if request.business_context else "",
                    industry=request.business_context.get("industry") if request.business_context else "",
                    target_audience=request.business_context.get("target_audience") if request.business_context else "",
                    current_revenue=request.business_context.get("current_revenue") if request.business_context else "",
                    main_challenges=request.business_context.get("main_challenges") if request.business_context else [],
                    goals=request.business_context.get("goals") if request.business_context else []
                )

                if request.force_system_nexus and request.nexus_tool_target:
                    logger.info("Forcing System Nexus tool call via /chat endpoint.", tool=request.nexus_tool_target)
                    routing_decision = {
                        "intent_type": "external_tool_call",
                        "primary_capability": "external_integration",
                        "recommended_agents": ["System Nexus"],
                        "workflow_type": "single",
                        "complexity_score": 5,
                        "estimated_time": "minutes",
                        "context_requirements": ["tool_specific"],
                        "reasoning": f"Forced tool call via ChatRequest for {request.nexus_tool_target}."
                    }
                    nexus_message = (
                        f"Perform the following action using the specified tool: "
                        f"Tool: {request.nexus_tool_target}, Arguments: {json.dumps(request.business_context)}. "
                        f"Reason: API forced call."
                        f"\nOriginal message context: {request.message}"
                    )
                    orchestration_result = await self.elite_agent_center.execute_workflow(
                        nexus_message,
                        routing_decision,
                        user_agent_context
                    )
                    
                    response_text = ""
                    if orchestration_result["success"] and "System Nexus" in orchestration_result["results"]:
                        nexus_output = orchestration_result["results"]["System Nexus"].output
                        response_text = f"Tool Execution Result: {nexus_output.get('response', 'Unknown result')}"
                        if nexus_output.get("tool_output"):
                            response_text += f"\n\nTool Output: ```json\n{json.dumps(nexus_output['tool_output'], indent=2)}\n```"

                    chat_response = ChatResponse(
                        success=orchestration_result["success"],
                        response=response_text,
                        model="System Nexus",
                        provider="Elite Command Center",
                        cost=orchestration_result["total_cost"],
                        response_time=orchestration_result["total_execution_time"],
                        reasoning=routing_decision["reasoning"],
                        agents_used=routing_decision["recommended_agents"],
                        workflow_type=routing_decision["workflow_type"],
                        confidence_score=orchestration_result["results"]["System Nexus"].confidence_score if "System Nexus" in orchestration_result["results"] else 0.0,
                        tool_calls_made=orchestration_result["tool_calls"]
                    )
                    background_tasks.add_task(self._save_chat_history, request.user_id, request.message, chat_response, orchestration_result["context_updates"])
                    return chat_response

                elif self.config.get("elite_agents.enabled", True) and request.task_type == "auto":
                    elite_decision = await self.ai_service.should_use_elite_agents(
                        request.message,
                        self.config.get("elite_agents.auto_routing_threshold", 0.7)
                    )

                    if elite_decision.get("use_elite_agents", False) and self.elite_agent_center:
                        logger.info("Routing to Elite Agents via auto-detection.", decision=elite_decision)
                        routing_decision = await self.intent_classifier.analyze_intent(
                            request.message,
                            user_agent_context
                        )
                        
                        orchestration_result = await self.elite_agent_center.execute_workflow(
                            request.message,
                            routing_decision,
                            user_agent_context
                        )

                        if orchestration_result["success"]:
                            response_parts = []
                            total_confidence = 0.0
                            agents_contributing = 0

                            for agent_name, agent_result in orchestration_result["results"].items():
                                if agent_result.success:
                                    agent_output_text = agent_result.output.get('response') or json.dumps(agent_result.output, indent=2)
                                    response_parts.append(f"**{agent_name}:**\n{agent_output_text}")
                                    total_confidence += agent_result.confidence_score
                                    agents_contributing += 1
                                else:
                                    response_parts.append(f"**{agent_name} (Failed):**\nError: {agent_result.output.get('error', 'Unknown Error')}")

                            avg_confidence = total_confidence / agents_contributing if agents_contributing > 0 else 0.0
                            combined_response = "\n\n".join(response_parts)

                            chat_response = ChatResponse(
                                success=True,
                                response=combined_response,
                                model="Elite Agent Team",
                                provider="Elite Command Center",
                                cost=orchestration_result["total_cost"],
                                response_time=orchestration_result["total_execution_time"],
                                reasoning=f"Elite routing: {elite_decision.get('reasoning', 'High complexity detected')}",
                                agents_used=orchestration_result["routing_decision"].get("recommended_agents", []),
                                workflow_type=orchestration_result["workflow_type"],
                                confidence_score=avg_confidence,
                                tool_calls_made=orchestration_result["tool_calls"]
                            )
                            background_tasks.add_task(self._save_chat_history, request.user_id, request.message, chat_response, orchestration_result["context_updates"])
                            return chat_response
                        else:
                            if self.config.get("elite_agents.fallback_to_standard", True):
                                logger.warning("Elite agent orchestration failed, falling back to standard model.", error=orchestration_result.get("error"))
                                pass
                            else:
                                raise HTTPException(status_code=500, detail=orchestration_result.get("error", "Elite agent orchestration failed and fallback disabled."))
                    else:
                        logger.info("Not routing to Elite Agents (auto-detection criteria not met or disabled).", decision=elite_decision)
                        pass

                final_task_type = request.task_type
                if final_task_type == "auto":
                    final_task_type = await self.ai_service.detect_task_type(
                        request.message,
                        self.config.get('classifier_model'),
                        self.config.get('valid_task_types', [])
                    )

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
                background_tasks.add_task(self._save_chat_history, request.user_id, request.message, chat_response_standard, {})
                return chat_response_standard

            except ValidationError as e:
                logger.error("Pydantic validation error for ChatRequest", exc_info=True, errors=e.errors())
                raise HTTPException(status_code=422, detail=f"Invalid request payload: {e.errors()}")
            except Exception as e:
                logger.error("Unhandled error in enhanced_chat_endpoint", exc_info=True)
                raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

        # ============================================================================
        # FILE HANDLING & LEGACY ENDPOINTS
        # ============================================================================

        @self.app.post("/upload-file", tags=["File Handling"])
        @handle_errors
        async def upload_file(file: UploadFile = File(...)):
            """Handles file uploads to the 'uploads' directory."""
            upload_dir = "uploads"
            filename = os.path.basename(file.filename)
            file_path = os.path.join(upload_dir, filename)
            
            counter = 1
            original_filename_base, original_filename_ext = os.path.splitext(filename)
            while os.path.exists(file_path):
                filename = f"{original_filename_base}_{counter}{original_filename_ext}"
                file_path = os.path.join(upload_dir, filename)
                counter += 1

            try:
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                logger.info("File uploaded successfully", filename=filename, file_path=file_path, user_id="anonymous")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "filename": filename,
                        "detail": f"File '{filename}' uploaded successfully.",
                        "file_path": file_path
                    }
                )
            except Exception as e:
                logger.error("Failed to upload file", filename=file.filename, error=str(e), exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


        @self.app.post("/delegate/marketing-copy", tags=["Legacy"], response_model=EliteAgentResponse)
        @handle_errors
        async def delegate_marketing_copy(topic: str = Form(...), user_id: str = Form("legacy_user")):
            """Legacy endpoint now routes to Conversion Predator via elite agents."""
            if not self.elite_agent_center:
                raise HTTPException(status_code=503, detail="Elite Agent Command Center not available.")
            
            logger.info("Legacy marketing copy request received, routing to Conversion Predator.")
            
            elite_request = EliteAgentRequest(
                message=f"Create killer marketing copy for: {topic}",
                user_id=user_id,
                business_type="unknown",
                industry="marketing",
                target_audience="customers"
            )
            
            routing_decision_simulated = {
                "intent_type": "single_agent",
                "primary_capability": "conversion_optimization",
                "recommended_agents": ["Conversion Predator"],
                "workflow_type": "single",
                "complexity_score": 5,
                "estimated_time": "minutes",
                "reasoning": "Legacy endpoint redirecting to Conversion Predator."
            }

            agent_context_from_legacy = AgentContext(
                business_type="unknown", industry="marketing", target_audience="customers"
            )
            
            orchestration_result = await self.elite_agent_center.execute_workflow(
                elite_request.message,
                routing_decision_simulated,
                agent_context_from_legacy
            )
            
            if orchestration_result["success"] and "Conversion Predator" in orchestration_result["results"]:
                conv_pred_result = orchestration_result["results"]["Conversion Predator"]
                return EliteAgentResponse(
                    success=True,
                    routing_decision=routing_decision_simulated,
                    workflow_result=orchestration_result,
                    execution_time=orchestration_result["total_execution_time"],
                    agents_used=["Conversion Predator"],
                    tool_calls_made=orchestration_result["tool_calls"]
                )
            else:
                raise HTTPException(status_code=500, detail=orchestration_result.get("error", "Failed to get marketing copy from Conversion Predator."))


        @self.app.post("/develop-feature", tags=["Legacy"], response_model=EliteAgentResponse)
        @handle_errors
        async def develop_feature(task: DevelopmentTask, user_id: str = Form("legacy_dev_user")):
            """Legacy endpoint now routes to Code Overlord via elite agents."""
            if not self.elite_agent_center:
                raise HTTPException(status_code=503, detail="Elite Agent Command Center not available.")
            
            logger.info("Legacy develop feature request received, routing to Code Overlord.")
            
            elite_request = EliteAgentRequest(
                message=f"Develop feature: {task.task_description}",
                user_id=user_id,
                business_type="software_development",
                industry="tech",
                main_challenges=["technical implementation"]
            )

            routing_decision_simulated = {
                "intent_type": "single_agent",
                "primary_capability": "code_optimization",
                "recommended_agents": ["Code Overlord"],
                "workflow_type": "single",
                "complexity_score": 7,
                "estimated_time": "hours",
                "reasoning": "Legacy endpoint redirecting to Code Overlord for development task."
            }
            
            agent_context_from_legacy = AgentContext(
                business_type="software_development", industry="tech", main_challenges=["technical implementation"]
            )

            orchestration_result = await self.elite_agent_center.execute_workflow(
                elite_request.message,
                routing_decision_simulated,
                agent_context_from_legacy
            )

            if orchestration_result["success"] and "Code Overlord" in orchestration_result["results"]:
                code_overlord_result = orchestration_result["results"]["Code Overlord"]
                return EliteAgentResponse(
                    success=True,
                    routing_decision=routing_decision_simulated,
                    workflow_result=orchestration_result,
                    execution_time=orchestration_result["total_execution_time"],
                    agents_used=["Code Overlord"],
                    tool_calls_made=orchestration_result["tool_calls"]
                )
            else:
                raise HTTPException(status_code=500, detail=orchestration_result.get("error", "Failed to get development plan from Code Overlord."))


    def _save_chat_history(self, user_id: str, message: str, response: ChatResponse, context_updates: Dict[str, Any]):
        """Background task to save chat history and context updates."""
        try:
            with self.DbSession() as session:
                history = ChatHistory(
                    user_id=user_id,
                    message=message,
                    response=response.response,
                    model_used=response.model,
                    provider=response.provider,
                    cost=response.cost,
                    response_time=response.response_time,
                    agents_used=json.dumps(response.agents_used),
                    workflow_type=response.workflow_type,
                    confidence_score=response.confidence_score,
                    tool_calls=json.dumps(response.tool_calls_made),
                    context_updates_json=json.dumps(context_updates)
                )
                session.add(history)
                session.commit()
                logger.info("Chat history saved successfully", user_id=user_id, model=response.model)
        except Exception as e:
            logger.error("Failed to save chat history", user_id=user_id, error=str(e), exc_info=True)
            if 'session' in locals():
                session.rollback()

    def run(self):
        """Runs the FastAPI server using Uvicorn."""
        host = self.config.get("api.host", "0.0.0.0")
        port = self.config.get("api.port", 8000)
        
        logger.info(f"🚀 Starting Elite Agent Command Center on http://{host}:{port}")
        logger.info(f"🔥 Elite Agents: {'Active' if self.elite_agent_center and len(self.elite_agent_center.agents) > 0 else 'Offline'}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level=self.config.get("logging.level", "info").lower(),
            access_log=True
        )

portal = EnhancedUnifiedAIPortal()
app = portal.app

if __name__ == "__main__":
    try:
        portal.run()
    except Exception as e:
        logger.critical(f"Failed to start Elite Agent Command Center: {e}", exc_info=True)