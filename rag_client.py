import asyncio
import uuid
from typing import AsyncGenerator
import pyttsx3  # Add TTS import

import streamlit as st
import ollama
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


# =========================
# Ollama + MCP Client
# =========================
class OllamaMCPClient:
    def __init__(self, model: str, server_url: str):
        self.model = model
        self.server_url = server_url
        self.messages: list[dict] = []
        self.available_tools: list[dict] = []
        
        # MODIFIED: Updated system prompt to distinguish voice commands
        self.system_prompt = (
            "You are an AI assistant that can use MCP tools exposed by the server.\n\n"
            
            "IMPORTANT COMMAND TYPES:\n"
            "1. VOICE COMMANDS (VCC): Commands starting with '/vc ' are voice commands "
            "that should be processed by the 'correct_command' tool to convert natural language "
            "to structured VUI commands (e.g., 'CLICK home', 'OPEN calculator').\n\n"
            
            "2. ECHO COMMANDS: Commands starting with '/echo ' are for immediate text-to-speech "
            "output and should NOT use any tools.\n\n"
            
            "3. REGULAR TOOLS: For other requests, use the available tools:\n"
            "- Arithmetic operations (add, subtract, multiply, divide)\n"
            "- Wikipedia scraping and querying (scrape_wikipedia, query_knowledge)\n"
            "- File operations (if available)\n"
            "- Terminal commands (if available)\n\n"
            
            "TOOL USAGE RULES:\n"
            "- Only call tools by their exact names from the provided tool list.\n"
            "- Do not invent tool names.\n"
            "- When you call a tool, provide valid arguments matching its schema.\n"
            "- For voice commands (/vc), ALWAYS use the 'correct_command' tool.\n"
            "- For echo commands (/echo), NEVER use any tools - just output the text.\n"
            "- After any tool result, explain the result clearly to the user.\n"
        )
        
        # ADDED: TTS engine initialization
        self.tts_engine = None
        self._initialize_tts()

    def _initialize_tts(self):
        """Initialize the text-to-speech engine."""
        try:
            self.tts_engine = pyttsx3.init()
            # Configure TTS settings
            self.tts_engine.setProperty('rate', 150)  # Speaking speed
            self.tts_engine.setProperty('volume', 0.8)  # Volume level
            print("✅ TTS engine initialized successfully")
        except Exception as e:
            print(f"❌ TTS initialization failed: {e}")
            self.tts_engine = None

    # ADDED: TTS component for reading out voice commands
    def text_to_speech(self, text: str):
        """Convert text to speech using pyttsx3.
        
        Args:
            text: The text to be spoken aloud
        """
        if not self.tts_engine:
            print("TTS engine not available")
            return
            
        try:
            # Clean the text - remove any special tokens or formatting
            clean_text = text.replace('<end_of_turn>', '').replace('<start_of_turn>', '').strip()
            
            # Only speak if it's a valid command (not empty or error messages)
            if (clean_text and 
                len(clean_text) > 0 and 
                not clean_text.startswith("Error:") and
                not clean_text.startswith("COMMAND NOT RECOGNIZED")):
                
                print(f"🔊 Speaking: {clean_text}")
                self.tts_engine.say(clean_text)
                self.tts_engine.runAndWait()
            else:
                print(f"⚠️ Not speaking invalid command: {clean_text}")
                
        except Exception as e:
            print(f"❌ TTS Error: {e}")

    # ADDED: Voice command processing method
    async def correct_voice_command(self, natural_command: str) -> str:
        """Process voice commands through the VCC pipeline.
        
        Args:
            natural_command: The natural language voice command
            
        Returns:
            The cleaned up structured command
        """
        try:
            async with streamablehttp_client(url=self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Call the correct_command tool
                    result = await session.call_tool("correct_command", {"query": natural_command})
                    
                    if result.content and len(result.content) > 0 and getattr(result.content[0], "text", None):
                        cleaned_command = result.content[0].text
                        
                        # ADDED: Read out the cleaned command using TTS
                        self.text_to_speech(cleaned_command)
                        
                        return cleaned_command
                    else:
                        return "COMMAND NOT RECOGNIZED"
                        
        except Exception as e:
            error_msg = f"Error processing voice command: {e}"
            print(error_msg)
            return error_msg

    async def initialize_tools(self):
        """Initialize connection and fetch tools (one-time operation)."""
        try:
            async with streamablehttp_client(url=self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    response = await session.list_tools()
                    self.available_tools = []
                    for tool in response.tools:
                        self.available_tools.append({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema,
                            },
                        })

            self.messages = [{"role": "system", "content": self.system_prompt}]
            return True, f"Connected! Tools available: {len(self.available_tools)}"
        except Exception as e:
            return False, f"Connection failed: {e}"

    def _execute_tool_sync(self, tool_call) -> str:
        """Execute tool synchronously by creating a new event loop."""

        async def _do_execute():
            async with streamablehttp_client(url=self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments or {}
                    result = await session.call_tool(tool_name, tool_args)
                    if result.content and len(result.content) > 0 and getattr(result.content[0], "text", None):
                        return result.content[0].text
                    return "Tool executed but returned no content."

        # Create new event loop for each tool execution
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_do_execute())
        finally:
            loop.close()

    def chat_stream(self, user_text: str, max_tool_turns: int = 4):
        """
        Generator that streams responses (converted from async to sync).
        This avoids the async context manager issue with st.write_stream.
        """
        self.messages.append({"role": "user", "content": user_text})

        for _ in range(max_tool_turns):
            stream = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=self.available_tools,
                stream=True,
            )

            assistant_text = ""
            tool_calls = []

            # Stream tokens as they arrive
            for chunk in stream:
                if chunk.message.content:
                    assistant_text += chunk.message.content
                    yield chunk.message.content
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)

            self.messages.append(
                {"role": "assistant", "content": assistant_text, "tool_calls": tool_calls or []}
            )

            if not tool_calls:
                return

            # Execute tools synchronously
            yield f"\n\n🔧 Executing {len(tool_calls)} tool(s)...\n\n"

            for tc in tool_calls:
                yield f"🔹 Running: {tc.function.name}\n"
                try:
                    raw = self._execute_tool_sync(tc)
                    tool_call_id = str(uuid.uuid4())

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": (
                            f"The tool '{tc.function.name}' has finished executing.\n"
                            f"Raw output:\n{raw}\n\n"
                            "Now explain this result to the user in a clear, human-readable way."
                        ),
                    })
                except Exception as e:
                    yield f"❌ Tool error: {e}\n"

            # Stream follow-up explanation
            stream2 = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=self.available_tools,
                stream=True,
            )

            followup_text = ""
            for chunk in stream2:
                if chunk.message.content:
                    followup_text += chunk.message.content
                    yield chunk.message.content

            self.messages.append({"role": "assistant", "content": followup_text})

            if not getattr(chunk.message, "tool_calls", None):
                return

        yield "\n\n⚠️ Couldn't complete within allowed tool-call steps."