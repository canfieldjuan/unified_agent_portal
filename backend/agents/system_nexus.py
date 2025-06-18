# backend/agents/system_nexus.py

import json
import openai
import time
import os
import asyncio
from typing import Dict, List, Any, Callable, Optional

from backend.agents.base_agent import BaseKillerAgent, AgentCapability, AgentContext, AgentResult, AgentStatus
import structlog

logger = structlog.get_logger()

class SystemNexus(BaseKillerAgent):
    """
    The System Nexus: A specialized agent for intelligent integration with external systems.
    Acts as a gateway to various APIs and tools (CRM, Code Repos, Marketing Platforms etc.).
    """

    def __init__(self, openai_client: Optional[openai.AsyncOpenAI]):
        super().__init__(
            "System Nexus",
            [AgentCapability.EXTERNAL_INTEGRATION],
            openai_client
        )
        self.tool_registry: Dict[str, Dict[str, Any]] = self._build_tool_registry()

    def _build_system_prompt(self) -> str:
        """
        Defines the System Nexus's role and lists the tools it has access to.
        This prompt is crucial for the LLM to understand which tool to "call".
        """
        tool_descriptions = "\n".join([
            f"- **{name}**: {details['description']}. Parameters: {details['parameters_description']}"
            for name, details in self.tool_registry.items()
        ])

        return f"""
You are the System Nexus, the master integrator and operational arm of the Elite Agent Command Center.
Your sole purpose is to execute commands by intelligently interacting with external systems and APIs.

You have access to the following specialized tools. You MUST choose the most appropriate tool to fulfill the request.

AVAILABLE TOOLS:
{tool_descriptions}

INSTRUCTION:
When a user asks you to perform an action that requires an external tool, respond ONLY with a JSON object.
The JSON object MUST follow this structure:
{{
  "tool_name": "Name of the tool to call",
  "tool_args": {{
    "parameter1": "value1",
    "parameter2": "value2",
    ...
  }},
  "reasoning": "Brief explanation of why this tool was chosen and how arguments were derived."
}}

If the request cannot be fulfilled by any available tool, or if more information is needed,
respond with a clear natural language question or explanation of what is missing.
DO NOT make up tool names or parameters.
"""

    def _build_tool_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Registers all external tools/APIs that the System Nexus can interact with.
        Each tool has a name, description, and parameter schema for the LLM.
        """
        return {
            "Google Cloud Application Integration: Trigger Workflow": {
                "description": "Triggers a specific workflow within Google Cloud Application Integration. Useful for automating tasks across Google Cloud services or external SaaS applications connected via App Integration.",
                "parameters_description": "workflow_id (string, required), payload (JSON object, optional) - the data to pass to the workflow.",
                "function": self._call_google_cloud_app_integration
            },
            "Knit AI: Automate Task": {
                "description": "Executes a defined automation or fetches data via Knit AI's unified API. Connects to CRM, marketing, and custom apps.",
                "parameters_description": "task_name (string, required) - e.g., 'UpdateCRM', 'FetchMarketingData'; params (JSON object, optional) - specific parameters for the Knit AI task.",
                "function": self._call_knit_ai
            },
            "n8n: Execute Workflow": {
                "description": "Triggers a specific n8n workflow. Ideal for complex multi-step automations across various web services.",
                "parameters_description": "webhook_url (string, required) - the unique webhook URL for the n8n workflow; data (JSON object, optional) - data to send to the webhook.",
                "function": self._call_n8n_workflow
            },
            "CodeGPT: Generate/Refactor Code": {
                "description": "Interfaces with CodeGPT for advanced code generation, refactoring, or analysis. Best for direct code manipulation tasks.",
                "parameters_description": "operation (string, required) - e.g., 'generate', 'refactor', 'debug'; code_context (string, required) - the code to operate on; instructions (string, required) - specific instructions for the operation.",
                "function": self._call_codegpt
            },
            "Relevance AI: Agent Action": {
                "description": "Invokes an action or agent within the Relevance AI platform. Can be used for content generation, research, or data processing tasks configured in Relevance AI.",
                "parameters_description": "action_name (string, required) - name of the action/agent; inputs (JSON object, optional) - input parameters for the Relevance AI action.",
                "function": self._call_relevance_ai
            },
            "File System: Read File": {
                "description": "Reads the content of a specified file from the local file system (e.g., from the 'uploads' directory).",
                "parameters_description": "file_path (string, required) - path to the file.",
                "function": self._read_local_file
            },
            "File System: Write File": {
                "description": "Writes content to a specified file on the local file system (e.g., to the 'output_files' directory).",
                "parameters_description": "file_path (string, required) - path to the file; content (string, required) - content to write.",
                "function": self._write_local_file
            },
        }

    async def _parse_tool_call(self, llm_response_content: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to parse the LLM's response as a tool call JSON.
        Returns the parsed dictionary or None if not a valid tool call.
        """
        try:
            parsed_json = json.loads(llm_response_content)
            if "tool_name" in parsed_json and "tool_args" in parsed_json:
                return parsed_json
            return None
        except json.JSONDecodeError:
            return None

    async def execute(self, request: str, context: AgentContext) -> AgentResult:
        """
        Executes the System Nexus's task by identifying and calling external tools.
        """
        start_time = time.time()
        self.status = AgentStatus.BUSY
        tool_calls_made = []

        try:
            system_prompt = self._build_system_prompt()
            enhanced_request = self._enhance_request_with_context(request, context)

            llm_response_content, llm_cost = await self._call_openai(
                system_prompt,
                enhanced_request,
                model="gpt-4-turbo-preview"
            )
            
            tool_call_request = await self._parse_tool_call(llm_response_content)
            
            if tool_call_request:
                tool_name = tool_call_request.get("tool_name")
                tool_args = tool_call_request.get("tool_args", {})
                reasoning = tool_call_request.get("reasoning", "No specific reasoning provided.")
                
                if tool_name in self.tool_registry:
                    tool_function = self.tool_registry[tool_name]["function"]
                    logger.info("System Nexus executing tool", tool=tool_name, args=tool_args, user_request_preview=request[:100])
                    
                    try:
                        tool_output = await tool_function(**tool_args)
                        success = True
                        output = {"response": f"Successfully executed tool '{tool_name}'. Output: {tool_output.get('message', str(tool_output))}", "tool_output": tool_output, "tool_name": tool_name, "reasoning": reasoning}
                        tool_calls_made.append({"tool_name": tool_name, "tool_args": tool_args, "status": "success", "output": tool_output})
                    except Exception as tool_exc:
                        success = False
                        tool_output = {"error": str(tool_exc), "message": f"Tool execution failed for {tool_name}."}
                        output = {"response": f"Failed to execute tool '{tool_name}'. Error: {str(tool_exc)}", "tool_output": tool_output, "tool_name": tool_name, "reasoning": reasoning}
                        tool_calls_made.append({"tool_name": tool_name, "tool_args": tool_args, "status": "failed", "error": str(tool_exc)})
                        logger.error("System Nexus tool execution failed", tool=tool_name, error=str(tool_exc), exc_info=True)
                else:
                    success = False
                    output = {"response": f"Error: Tool '{tool_name}' not found in registry. Reasoning: {reasoning}", "tool_name": tool_name, "reasoning": reasoning}
                    logger.warning("System Nexus tried to call unregistered tool", tool=tool_name, user_request_preview=request[:100])
            else:
                success = False
                output = {"response": f"System Nexus could not identify a clear tool call. LLM output was:\n\n```json\n{llm_response_content}\n```\n\nPlease refine your request if you intended a tool interaction, or clarify what you need.",
                          "raw_llm_output": llm_response_content,
                          "reasoning": "LLM did not output a valid tool call JSON."}
                logger.warning("System Nexus could not parse LLM output as tool call", llm_output=llm_response_content[:200])

            execution_time = time.time() - start_time
            self.status = AgentStatus.IDLE
            
            total_cost = llm_cost
            confidence = self._calculate_confidence(output)
            
            self._update_metrics(execution_time, total_cost, success, confidence)

            return AgentResult(
                agent_name=self.name,
                success=success,
                output=output,
                execution_time=execution_time,
                cost=total_cost,
                confidence_score=confidence,
                tool_calls=tool_calls_made
            )

        except Exception as e:
            self.status = AgentStatus.ERROR
            execution_time = time.time() - start_time
            self._update_metrics(execution_time, 0.0, False, 0.0)
            logger.error("System Nexus failed to execute due to unhandled error", error=str(e), exc_info=True)
            return AgentResult(
                agent_name=self.name,
                success=False,
                output={"error": str(e), "message": "System Nexus encountered an unhandled error during execution."},
                execution_time=execution_time,
                cost=0.0,
                confidence_score=0.0,
                tool_calls=tool_calls_made
            )

    # ========================================================================
    # PLACEHOLDER TOOL INTEGRATION METHODS (YOU FILL THESE IN)
    # ========================================================================

    async def _call_google_cloud_app_integration(self, workflow_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Placeholder: Simulates triggering a Google Cloud Application Integration workflow.
        """
        logger.info("Calling Google Cloud Application Integration workflow (simulated)", workflow_id=workflow_id, payload=payload)
        await asyncio.sleep(2)
        return {"status": "success", "message": f"GC App Integration workflow '{workflow_id}' triggered (simulated).", "response_data": {"workflow_status": "started", "payload_received": payload}}

    async def _call_knit_ai(self, task_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Placeholder: Simulates executing a task via Knit AI.
        """
        logger.info("Calling Knit AI task (simulated)", task_name=task_name, params=params)
        await asyncio.sleep(1.5)
        return {"status": "success", "message": f"Knit AI task '{task_name}' executed (simulated).", "response_data": {"task_result": "processed", "data_sent": params}}

    async def _call_n8n_workflow(self, webhook_url: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Placeholder: Simulates triggering an n8n webhook workflow.
        """
        logger.info("Calling n8n workflow webhook (simulated)", webhook_url=webhook_url, data=data)
        await asyncio.sleep(1)
        return {"status": "success", "message": f"n8n workflow triggered at {webhook_url} (simulated).", "response_data": {"webhook_received": True, "data_sent": data}}

    async def _call_codegpt(self, operation: str, code_context: str, instructions: str) -> Dict[str, Any]:
        """
        Placeholder: Simulates interacting with CodeGPT for code generation/refactoring.
        """
        logger.info("Calling CodeGPT (simulated)", operation=operation, code_context_preview=code_context[:50], instructions=instructions)
        await asyncio.sleep(3)
        generated_code = f"// CodeGPT generated {operation} (simulated):\n/* {instructions} */\n{code_context.upper()}_PROCESSED_CODE;"
        return {"status": "success", "code_output": generated_code, "summary": f"Code generation/refactoring completed for '{operation}' (simulated)."}

    async def _call_relevance_ai(self, action_name: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Placeholder: Simulates invoking an agent or action within Relevance AI.
        """
        logger.info("Calling Relevance AI agent action (simulated)", action_name=action_name, inputs=inputs)
        await asyncio.sleep(2.5)
        return {"status": "success", "message": f"Relevance AI action '{action_name}' executed (simulated).", "relevance_ai_output": {"data": "some_generated_content_or_research_result"}}

    async def _read_local_file(self, file_path: str) -> Dict[str, Any]:
        """
        Reads content from a local file. (Restricted to 'uploads' for security).
        """
        filename_only = os.path.basename(file_path)
        safe_path = os.path.join("uploads", filename_only)
        
        if not os.path.exists(safe_path):
            raise FileNotFoundError(f"File not found: {filename_only}")
        if not os.path.isfile(safe_path):
            raise ValueError(f"Path is not a file: {filename_only}")
        
        try:
            with open(safe_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info("Read local file successfully", file=filename_only)
            return {"file_content": content, "filename": filename_only, "status": "success"}
        except Exception as e:
            logger.error("Failed to read local file", file=filename_only, error=str(e), exc_info=True)
            raise RuntimeError(f"Failed to read file '{filename_only}': {e}")

    async def _write_local_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Writes content to a local file. (Restricted to a 'output_files' directory for security).
        """
        filename_only = os.path.basename(file_path)
        output_dir = "output_files"
        os.makedirs(output_dir, exist_ok=True)
        safe_path = os.path.join(output_dir, filename_only)
        
        try:
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info("Wrote local file successfully", file=filename_only, path=safe_path)
            return {"status": "success", "message": f"Content written to {safe_path}.", "file_path": safe_path}
        except Exception as e:
            logger.error("Failed to write local file", file=filename_only, error=str(e), exc_info=True)
            raise RuntimeError(f"Failed to write file '{filename_only}': {e}")