# app.py
import streamlit as st
import os
import sys
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.memory import ChatMemoryBuffer

# --- 配置部分 ---
GOOGLE_API_KEY = "AIzaSyAfgLQT8ZsklX5Xxsk7Mdtyo2wLEf6VAj8"

# 页面设置
st.set_page_config(page_title="Now Agent", page_icon="🌿", layout="centered")
st.title("🌿 The Power of Now · 疗愈 Agent")

# --- 核心逻辑 ---

# 1. 配置模型 (必须要有，否则聊天时会不知道用谁)
# 注意：这里不需要再建立索引，只需要配置好让 LlamaIndex 知道用 Gemini 回答
Settings.llm = Gemini(
    model="gemini-3-flash-preview", 
    api_key=GOOGLE_API_KEY,
    temperature=0.3
)
Settings.embed_model = GeminiEmbedding(
    model_name="gemini-embedding-001", 
    api_key=GOOGLE_API_KEY
)

# 2. 加载索引 (只从硬盘读，不调 API，不花钱)
@st.cache_resource(show_spinner=False)
def load_index():
    persist_dir = "./storage"
    if not os.path.exists(persist_dir):
        st.error("❌ 找不到索引文件！请先运行 'python build_index.py' 来构建知识库。")
        return None
    
    with st.spinner("正在连接内在智慧 (从硬盘加载)..."):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        return index

# --- 导师人格设定 ---
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

    # --- 聊天界面 (和之前一样) ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("此刻，你感受到了什么？"):
        # 1. 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. 生成回答
        with st.chat_message("assistant"):
            # 获取响应对象 (这里面包含了 text 和 source_nodes)
            response = st.session_state.chat_engine.stream_chat(prompt)
            
            # 实时流式输出文字
            full_response = st.write_stream(response.response_gen)
            
            # --- 【新加的功能：显示引用来源】 ---
            # 检查是否有引用源 (有时候纯闲聊可能没有源)
            if hasattr(response, 'source_nodes') and response.source_nodes:
                # 使用 expander 折叠起来，保持界面极简
                with st.expander("📖 查看灵感来源 (来自《当下的力量》原文)"):
                    for node in response.source_nodes:
                        # 显示相似度分数 (Score) 和具体内容
                        # score 越高表示越相关
                        similarity = f"{node.score:.2f}" if node.score else "N/A"
                        st.markdown(f"**关联度:** `{similarity}`")
                        
                        # 显示切片原文
                        st.caption(node.text) 
                        st.divider() # 分割线

        # 3. 保存助手消息到历史记录
        st.session_state.messages.append({"role": "assistant", "content": full_response})