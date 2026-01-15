import streamlit as st
import asyncio
from rag_client import OllamaMCPClient

# =========================
# Streamlit UI
# =========================
def init_session_state():
    """Initialize session state variables."""
    if "client" not in st.session_state:
        st.session_state.client = None
    if "connected" not in st.session_state:
        st.session_state.connected = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "server_url" not in st.session_state:
        st.session_state.server_url = "http://127.0.0.1:3000/mcp"
    if "model_name" not in st.session_state:
        st.session_state.model_name = "llama3.2:3b"


async def connect_to_server(server_url: str, model: str):
    """Connect to MCP server and initialize tools."""
    client = OllamaMCPClient(model=model, server_url=server_url)
    success, message = await client.initialize_tools()
    return client, success, message


# ADDED: Minimal voice command detection
def is_voice_command(input_text: str) -> tuple[str, str]:
    """Check if input is a voice command and return command type and content."""
    input_text = input_text.strip()
    
    if input_text.startswith('/echo '):
        return 'echo', input_text[6:].strip()
    elif input_text.startswith('/vc '):
        return 'vc', input_text[4:].strip()
    else:
        return 'chat', input_text


# ADDED: Simple voice command processing
async def process_vc_command(content: str) -> str:
    """Process voice command asynchronously."""
    if st.session_state.client:
        return await st.session_state.client.correct_voice_command(content)
    return "Client not connected"


def main():
    st.set_page_config(page_title="MCP + Ollama Chat", page_icon="🤖", layout="wide")

    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        server_url = st.text_input(
            "MCP Server URL",
            value=st.session_state.server_url,
            help="URL of your MCP server"
        )

        model_name = st.text_input(
            "Ollama Model",
            value=st.session_state.model_name,
            help="Name of the Ollama model to use"
        )

        if st.button("Connect", type="primary", use_container_width=True):
            with st.spinner("Connecting to MCP server..."):
                client, success, message = asyncio.run(connect_to_server(server_url, model_name))
                if success:
                    st.session_state.client = client
                    st.session_state.connected = True
                    st.session_state.server_url = server_url
                    st.session_state.model_name = model_name
                    st.success(message)
                else:
                    st.error(message)

        if st.session_state.connected:
            st.success("✅ Connected")
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.client:
                    st.session_state.client.messages = [
                        {"role": "system", "content": st.session_state.client.system_prompt}
                    ]
                st.rerun()
        else:
            st.warning("⚠️ Not connected")

        st.divider()
        # UPDATED: Tips section with voice commands
        st.markdown("""
        ### 💡 Tips
        - Connect to your MCP server first
        - Use `/echo [text]` for immediate TTS
        - Use `/vc [command]` for voice command conversion  
        - Regular text uses tools automatically
        - Responses stream in real-time
        """)

    # Main chat interface
    st.title("🤖 MCP + Ollama Chat")
    # UPDATED: Added voice command info
    st.markdown("Chat with your AI assistant powered by Ollama and MCP tools. Use `/echo` or `/vc` for voice commands.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    # UPDATED: Added voice command hint
    if prompt := st.chat_input("Type your message or use /echo /vc commands...", disabled=not st.session_state.connected):
        if not st.session_state.connected:
            st.error("Please connect to the MCP server first!")
            return

        # ADDED: Voice command routing
        command_type, content = is_voice_command(prompt)
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Stream assistant response
        with st.chat_message("assistant"):
            try:
                if command_type == 'echo':
                    # Immediate TTS for echo commands
                    if st.session_state.client:
                        st.session_state.client.text_to_speech(content)
                    st.success(f"🔊 Spoke: '{content}'")
                    st.session_state.messages.append({"role": "assistant", "content": f"Echoed: {content}"})
                    
                elif command_type == 'vc':
                    # Process voice command through MCP pipeline
                    with st.spinner("🔄 Converting voice command..."):
                        corrected_command = asyncio.run(process_vc_command(content))
                        st.success(f"🎤 '{content}' → '{corrected_command}'")
                        st.session_state.messages.append({"role": "assistant", "content": f"Converted to: {corrected_command}"})
                        
                else:
                    # Normal chat processing
                    full_response = st.write_stream(
                        st.session_state.client.chat_stream(prompt)
                    )
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
            except Exception as e:
                error_msg = f"❌ Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()