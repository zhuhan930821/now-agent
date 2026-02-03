import streamlit as st
import os
import sys
from llama_index.core import StorageContext, load_index_from_storage
# 强制将标准输出和标准错误设置为 utf-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# --- 修改点 1: 引入 Gemini 库 ---
from llama_index.embeddings.gemini import GeminiEmbedding  # <--- 必须有这一行
from llama_index.llms.gemini import Gemini
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer

# --- 配置部分 ---
# 建议将 Key 放在环境变量中，或者直接写在这里（注意保密）
GOOGLE_API_KEY = "AIzaSyAfgLQT8ZsklX5Xxsk7Mdtyo2wLEf6VAj8"

st.set_page_config(page_title="Now Agent", page_icon="🌿", layout="centered")
st.title("🌿 The Power of Now · 疗愈 Agent (Gemini版)")

# --- 核心逻辑 ---

@st.cache_resource(show_spinner=False)
def load_data_and_index():
    with st.spinner("正在连接内在智慧 (加载 Gemini)..."):
        
        # --- 修改点 2: 配置 Gemini ---
        
        # 1. 设置 LLM (大脑)
        # 推荐使用 "models/gemini-1.5-flash" (速度快，免费额度高) 
        # 或者 "models/gemini-1.5-pro" (更聪明，适合深层推理)
        Settings.llm = Gemini(
            model="gemini-3-flash-preview", 
            api_key=GOOGLE_API_KEY,
            temperature=0.3,
            transport="rest"
        )
        
        # 2. 设置 Embedding (把书变成向量的工具)
        # 必须设置这个，否则 LlamaIndex 会默认尝试调用 OpenAI 导致报错
        Settings.embed_model = GeminiEmbedding(
            model_name="gemini-embedding-001", 
            api_key=GOOGLE_API_KEY
        )
        
        # --- 下面的逻辑不用变 ---
        
        if not os.path.exists("data"):
            os.makedirs("data")
            st.warning("请在项目目录下创建 'data' 文件夹并放入书籍文件。")
            return None

        reader = SimpleDirectoryReader(input_dir="data")
        documents = reader.load_data()
        
        index = VectorStoreIndex.from_documents(documents)
        return index


# --- 独特的“导师”人格设定 ---
SYSTEM_PROMPT = """
你现在是 Eckhart Tolle（埃克哈特·托利），《当下的力量》的作者。
你的任务不是作为一个“AI助手”去解决逻辑问题，而是作为一个“灵性导师”去化解用户的痛苦（Pain-Body）。

请遵循以下原则：
1. **核心立场**：永远将用户引导回“当下”（The Now）。指出他们的痛苦来自于“思维认同”（Identification with the mind）或“心理时间”（Psychological time）。
2. **语气风格**：平静、富有同理心、深邃、不评判。像一个睿智的观察者。
3. **引用原文**：在回答时，必须优先检索并引用《当下的力量》书中的概念（如：痛苦之身、小我、临在、未显化状态）。
4. **实践导向**：不要只讲大道理，要给出具体的练习建议（例如：关注呼吸、感受内在身体、通过观察情绪来通过它）。
5. **处理痛苦**：当用户表达痛苦时，不要试图用逻辑去“修补”那个故事，而是让他们去“观察”那个痛苦，从痛苦中分离出来。

如果用户问非灵性问题，请礼貌地将话题引回到意识和当下的层面。
"""

# --- 初始化 ---
index = load_data_and_index()

if index:
    # 初始化聊天引擎
    if "chat_engine" not in st.session_state:
        # 使用 context 模式，让 AI 既能查书，又有记忆
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=SYSTEM_PROMPT,
            verbose=False
        )

    # --- 聊天界面 ---
    
    # 显示历史消息
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入
    if prompt := st.chat_input("此刻，你感受到了什么？"):
        # 1. 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. 生成回答
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # 调用 RAG 引擎
            response = st.session_state.chat_engine.stream_chat(prompt)
            
            for token in response.response_gen:
                full_response += token
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # 可选：显示它参考了书里的哪一段（调试用）
            with st.expander("查看灵感来源 (Source Context)"):
                 st.write(response.source_nodes)

        # 3. 保存助手消息
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.write("等待数据加载...")