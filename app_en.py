import streamlit as st
import os
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.memory import ChatMemoryBuffer

# --- Page Config ---
st.set_page_config(page_title="Now Agent", page_icon="🌿", layout="centered")

# --- Security Check (Optional: Keep if you use password) ---
if "ACCESS_PASSWORD" in st.secrets:
    def check_password():
        if "password_correct" not in st.session_state:
            st.session_state.password_correct = False
        if st.session_state.password_correct:
            return True
        st.title("🔒 Please Enter Access Code")
        password_input = st.text_input("Access Code (my DOB)", type="password")
        if st.button("Enter The Now"):
            if password_input == st.secrets["ACCESS_PASSWORD"]:
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("Incorrect password. You remain trapped in the mind...")
        return False
    if not check_password():
        st.stop()

# --- Main UI ---
st.title("🌿 The Power of Now · Healing Agent")
st.caption("“Realize deeply that the present moment is all you have.”")

# --- Configuration ---
# Load API Key from Streamlit Secrets (Recommended)
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

Settings.llm = Gemini(
    model="gemini-3-flash-preview", 
    api_key=GOOGLE_API_KEY,
    temperature=0.3
)
Settings.embed_model = GeminiEmbedding(
    model_name="gemini-embedding-001", 
    api_key=GOOGLE_API_KEY
)

# --- Load Existing Index ---
@st.cache_resource(show_spinner=False)
def load_index():
    persist_dir = "./storage"
    if not os.path.exists(persist_dir):
        st.error("❌ Index not found! Please run build_index.py first.")
        return None
    
    with st.spinner("Connecting to inner wisdom..."):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        return index

# --- 🧠 English System Prompt (关键修改) ---
SYSTEM_PROMPT = """
You are Eckhart Tolle, the author of "The Power of Now".
Your task is not to be an AI assistant that solves logical problems, but a spiritual guide to help dissolve the user's "Pain-Body".

Please follow these principles:
1. **Core Stance**: Always guide the user back to "The Now". Point out that their suffering comes from "Identification with the mind" or "Psychological time".
2. **Tone**: Calm, compassionate, profound, non-judgmental. Speak like a wise observer.
3. **Use Quotes**: Prioritize retrieving and quoting concepts from the book (e.g., Pain-Body, Ego, Presence, Unmanifested).
4. **Practice Oriented**: Don't just give theories; give specific practices (e.g., "Feel the inner body", "Watch the thinker").
5. **Handling Pain**: When the user expresses pain, do not try to "fix" the story with logic. Ask them to "observe" the pain and disidentify from it.

Reply in English.
"""

# --- Chat Logic ---
index = load_index()

if index:
    if "chat_engine" not in st.session_state:
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=SYSTEM_PROMPT,
            verbose=False
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # UI Input in English
    if prompt := st.chat_input("What are you feeling right now?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.session_state.chat_engine.stream_chat(prompt)
            full_response = st.write_stream(response.response_gen)
            
            # Citation in English
            if hasattr(response, 'source_nodes') and response.source_nodes:
                with st.expander("📖 View Source Context (The Power of Now)"):
                    for node in response.source_nodes:
                        similarity = f"{node.score:.2f}" if node.score else "N/A"
                        st.markdown(f"**Similarity Score:** `{similarity}`")
                        st.caption(node.text)
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": full_response})