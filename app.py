import streamlit as st
import nest_asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

# Apply nest_asyncio fix
nest_asyncio.apply()

# Load API key
gemini_api_key = st.secrets["GEMINI_API_KEY"]

if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY is not set. Please check your .env file.")
    st.stop()

# Create OpenAI-compatible Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Create model object
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Create config for running the agent
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Create Writer Agent
writer_agent = Agent(
    name="✍️ Writer Agent",
    instructions="""
        You are a helpful writing assistant. You can write essays, poems, stories, emails,
        and more. Be creative, clear, and helpful.
    """
)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="✍️ Writer Agent", page_icon="✍️")
st.title("✍️ Writer Agent")
st.markdown("""
Welcome! I can help you write **essays**, **poems**, **emails**, **stories**, and more.  
🟢 Just type what you want me to write!
""")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_input = st.chat_input("Type your request here...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Placeholder for assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("⏳ Generating your response...")

    try:
        # Run agent synchronously (safe inside Streamlit with nest_asyncio)
        result = Runner.run_sync(
            writer_agent,
            input=user_input,
            run_config=config
        )

        # Show final output
        response_text = result.final_output
        response_placeholder.markdown(response_text)

        # Save to history
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    except Exception as e:
        response_placeholder.markdown(f"❌ Error: {str(e)}")
