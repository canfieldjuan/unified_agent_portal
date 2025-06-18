# backend/services/orchestration.py

import asyncio
import json
import time
from typing import Dict, List, Any, Union

from backend.agents.base_agent import BaseKillerAgent, AgentContext, AgentResult
from backend.agents.system_nexus import SystemNexus # Import the SystemNexus agent
import structlog

logger = structlog.get_logger()

class AgentOrchestrator:
    """
    Orchestrates multi-agent workflows based on intent classification.
    Manages sequential, parallel, and hybrid execution of agents,
    and routes to SystemNexus for external tool calls.
    """

    def __init__(self, agents: Dict[str, BaseKillerAgent]):
        self.agents = agents
        # Ensure SystemNexus is in the agents dictionary if it's supposed to be used
        if "System Nexus" not in self.agents:
            logger.warning("System Nexus agent not found in the agents dictionary during orchestrator initialization. It will not be available for routing.")

    async def execute_workflow(
        self,
        user_request: str,
        routing_decision: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Executes a coordinated agent workflow based on the routing decision.
        """
        workflow_type = routing_decision.get("workflow_type", "single")
        recommended_agents = routing_decision.get("recommended_agents", [])
        intent_type = routing_decision.get("intent_type", "single_agent")

        logger.info(
            "Executing workflow",
            workflow_type=workflow_type,
            intent_type=intent_type,
            recommended_agents=recommended_agents,
            user_request_preview=user_request[:50]
        )

        results = {}
        total_cost = 0.0
        total_execution_time = 0.0
        all_tool_calls = [] # Collect all tool calls made during the workflow
        current_context = context # Context that evolves during sequential workflows

        try:
            if intent_type == "external_tool_call":
                # Direct route to System Nexus for external tool interactions
                if "System Nexus" in self.agents:
                    logger.info("Routing to System Nexus for external tool call", user_request=user_request)
                    result = await self.agents["System Nexus"].execute(user_request, current_context)
                    results["System Nexus"] = result
                    total_cost += result.cost
                    total_execution_time = result.execution_time
                    current_context = self._merge_context(current_context, result.context_updates)
                    all_tool_calls.extend(result.tool_calls)
                else:
                    logger.error("Intent classified as external_tool_call but System Nexus agent not available.")
                    return {
                        "success": False,
                        "error": "System Nexus agent is required for external tool calls but is not initialized.",
                        "results": {},
                        "total_execution_time": 0.0,
                        "total_cost": 0.0,
                        "tool_calls": []
                    }
            elif workflow_type == "single" or intent_type == "single_agent":
                if not recommended_agents:
                    raise ValueError("No agents recommended for single agent workflow.")
                agent_name = recommended_agents[0]
                logger.info("Executing single agent workflow", agent_name=agent_name, user_request=user_request)
                result = await self._execute_single_agent(user_request, agent_name, current_context)
                results[agent_name] = result
                total_cost += result.cost
                total_execution_time = result.execution_time
                current_context = self._merge_context(current_context, result.context_updates)
                all_tool_calls.extend(result.tool_calls)

            elif workflow_type == "sequential":
                logger.info("Executing sequential workflow", agents=recommended_agents, user_request=user_request)
                results, total_cost, total_execution_time, updated_context, workflow_tool_calls = \
                    await self._execute_sequential_workflow(user_request, recommended_agents, current_context)
                current_context = updated_context
                all_tool_calls.extend(workflow_tool_calls)

            elif workflow_type == "parallel":
                logger.info("Executing parallel workflow", agents=recommended_agents, user_request=user_request)
                results, total_cost, total_execution_time, updated_context, workflow_tool_calls = \
                    await self._execute_parallel_workflow(user_request, recommended_agents, current_context)
                current_context = updated_context
                all_tool_calls.extend(workflow_tool_calls)

            elif workflow_type == "hybrid":
                logger.info("Executing hybrid workflow (as sequential fallback)", agents=recommended_agents, user_request=user_request)
                # For simplicity, hybrid is executed as sequential here.
                # In a more advanced system, this would involve a complex state machine or a supervisor agent.
                results, total_cost, total_execution_time, updated_context, workflow_tool_calls = \
                    await self._execute_sequential_workflow(user_request, recommended_agents, current_context)
                current_context = updated_context
                all_tool_calls.extend(workflow_tool_calls)

            else:
                logger.warning("Unknown workflow type, defaulting to single agent.", workflow_type=workflow_type)
                if not recommended_agents:
                    raise ValueError("No agents recommended for unknown workflow type.")
                agent_name = recommended_agents[0]
                result = await self._execute_single_agent(user_request, agent_name, current_context)
                results[agent_name] = result
                total_cost += result.cost
                total_execution_time = result.execution_time
                current_context = self._merge_context(current_context, result.context_updates)
                all_tool_calls.extend(result.tool_calls)

            return {
                "success": all(r.success for r in results.values()),
                "workflow_type": workflow_type,
                "results": {name: result for name, result in results.items()}, # Convert to dict
                "total_execution_time": total_execution_time,
                "total_cost": total_cost,
                "context_updates": asdict(current_context), # Return the final updated context as dict
                "tool_calls": all_tool_calls # Propagate tool calls
            }

        except Exception as e:
            logger.error("Error during workflow execution", exc_info=True, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "results": {},
                "total_execution_time": total_execution_time,
                "total_cost": total_cost,
                "context_updates": asdict(current_context),
                "tool_calls": all_tool_calls
            }

    async def _execute_single_agent(
        self,
        request: str,
        agent_name: str,
        context: AgentContext
    ) -> AgentResult:
        """Executes a single agent's task."""
        if agent_name not in self.agents:
            logger.error("Attempted to execute non-existent agent", agent_name=agent_name)
            return AgentResult(
                agent_name=agent_name,
                success=False,
                output={"error": f"Agent '{agent_name}' not found or initialized."},
                execution_time=0.0,
                cost=0.0,
                confidence_score=0.0
            )
        agent = self.agents[agent_name]
        return await agent.execute(request, context)

    async def _execute_sequential_workflow(
        self,
        request: str,
        agent_names: List[str],
        context: AgentContext
    ) -> tuple[Dict[str, AgentResult], float, float, AgentContext, List[Dict[str, Any]]]:
        """Executes agents in sequence, passing context between them."""
        results = {}
        total_cost = 0.0
        total_time = 0.0
        current_context = context
        workflow_tool_calls = []

        for i, agent_name in enumerate(agent_names):
            if agent_name not in self.agents:
                logger.warning("Agent not found in sequential workflow, skipping", agent_name=agent_name)
                continue

            agent = self.agents[agent_name]

            # The request for subsequent agents might be a refinement based on previous results
            # For now, it's the original request, but this is a key extension point.
            processed_request = request # Or enhance_request_with_previous_results(request, results)

            result = await agent.execute(processed_request, current_context)
            results[agent_name] = result

            total_cost += result.cost
            total_time += result.execution_time
            workflow_tool_calls.extend(result.tool_calls)

            # Update context for the next agent in the sequence
            if result.context_updates:
                current_context = self._merge_context(current_context, result.context_updates)
            
            # If any agent in a sequence fails, you might want to stop or continue based on policy
            if not result.success:
                logger.warning("Agent failed in sequential workflow, stopping sequence.", agent_name=agent_name)
                break # Stop if one agent fails

        return results, total_cost, total_time, current_context, workflow_tool_calls

    async def _execute_parallel_workflow(
        self,
        request: str,
        agent_names: List[str],
        context: AgentContext
    ) -> tuple[Dict[str, AgentResult], float, float, AgentContext, List[Dict[str, Any]]]:
        """Executes agents in parallel."""
        tasks = []
        agents_to_run = []
        for agent_name in agent_names:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                tasks.append(agent.execute(request, context))
                agents_to_run.append(agent_name)
            else:
                logger.warning("Agent not found for parallel workflow, skipping", agent_name=agent_name)

        # Run all selected agent tasks concurrently
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        total_cost = 0.0
        max_time = 0.0 # Max time for parallel is the time of the longest running task
        merged_context_updates = {}
        workflow_tool_calls = []

        for i, agent_name in enumerate(agents_to_run):
            result_or_exception = raw_results[i]
            if isinstance(result_or_exception, Exception):
                logger.error(f"Parallel agent '{agent_name}' failed with exception", error=str(result_or_exception))
                result = AgentResult(
                    agent_name=agent_name,
                    success=False,
                    output={"error": str(result_or_exception), "message": f"Parallel agent {agent_name} failed."},
                    execution_time=0.0, cost=0.0, confidence_score=0.0
                )
            else:
                result = result_or_exception # It's an AgentResult object
                total_cost += result.cost
                max_time = max(max_time, result.execution_time)
                merged_context_updates.update(result.context_updates) # Merge context updates from all parallel agents
                workflow_tool_calls.extend(result.tool_calls)

            results[agent_name] = result

        # The final context will be the original context merged with all parallel updates
        final_context = self._merge_context(context, merged_context_updates)

        return results, total_cost, max_time, final_context, workflow_tool_calls
    
    def _merge_context(self, original: AgentContext, updates: Dict[str, Any]) -> AgentContext:
        """Merges updates into the AgentContext instance."""
        # Convert dataclass to dict, update, then convert back.
        # This handles nested fields like lists/dicts by replacing them if a new value is provided.
        merged_dict = asdict(original)
        
        for key, value in updates.items():
            if isinstance(value, list) and key in merged_dict and isinstance(merged_dict[key], list):
                # For lists, extend or replace if explicit (e.g., challenges, goals)
                merged_dict[key].extend(v for v in value if v not in merged_dict[key]) # Add unique items
            elif isinstance(value, dict) and key in merged_dict and isinstance(merged_dict[key], dict):
                # For dictionaries, deep merge
                merged_dict[key].update(value)
            else:
                # For other types, just replace
                merged_dict[key] = value
        
        # Reconstruct AgentContext from the merged dictionary
        return AgentContext(**merged_dict)