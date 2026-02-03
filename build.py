# build_index.py
import os
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# --- 1. 配置 Gemini (把你的 Key 填在这里) ---
GOOGLE_API_KEY = "AIzaSyAfgLQT8ZsklX5Xxsk7Mdtyo2wLEf6VAj8"

# 强制设置 UTF-8 (防止 Windows 乱码)
sys.stdout.reconfigure(encoding='utf-8')

print("🚀 开始初始化模型配置...")

# 设置大脑 (LLM)
Settings.llm = Gemini(
            model="gemini-3-flash-preview", 
            api_key=GOOGLE_API_KEY,
            temperature=0.3,
            transport="rest"
        )
        

# 设置翻译器 (Embedding)
Settings.embed_model = GeminiEmbedding(
            model_name="gemini-embedding-001", 
            api_key=GOOGLE_API_KEY
        )
        

# --- 2. 核心构建逻辑 ---
def build_and_save():
    # 检查 data 文件夹
    if not os.path.exists("data"):
        print("❌ 错误：找不到 'data' 文件夹。请创建并放入书籍文件。")
        return

    print("📚 正在读取 data 文件夹中的书籍...")
    reader = SimpleDirectoryReader(input_dir="data")
    documents = reader.load_data()
    print(f"✅ 读取成功！共找到 {len(documents)} 个文档片段。")

    print("🧠 正在发送给 Gemini 进行向量化 (这可能需要一点时间，请耐心等待)...")
    # 这一步会消耗 API 配额
    index = VectorStoreIndex.from_documents(documents)
    
    print("💾 正在将索引保存到硬盘 (storage 文件夹)...")
    index.storage_context.persist(persist_dir="./storage")
    print("🎉 大功告成！索引已构建完毕。现在你可以直接运行 app.py 了。")

if __name__ == "__main__":
    build_and_save()