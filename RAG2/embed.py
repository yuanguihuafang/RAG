# Embedding与向量检索模块
"""
本模块负责：
1. 文本向量化：将文本转换为向量表示
2. 向量存储：将向量存储到 ChromaDB 向量数据库
3. 向量检索：根据用户查询检索最相关的文本块
4. 生成回答：调用大语言模型生成最终回答
"""
import chunk
import chromadb 
import dashscope            # 导入阿里云的 dashscope 整个库。
from dashscope import (
    TextEmbedding,
)                           # 导入TextEmbedding类，用于调用DashScope的文本嵌入模型
from http import HTTPStatus # 导入HTTPStatus类，用于表示HTTP状态码
import os 

# from google import genai
# # 导入genai库，用于调用Google的Gemini模型
# google_client = genai.Client()
# EMBEDDING_MODEL = "gemini-embedding-exp-03-07"
# LLM_MODEL = "gemini-2.5-flash-preview-05-20"

from dotenv import load_dotenv 

load_dotenv()  # 加载.env文件中的环境变量

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")  # 配置千问 API Key（建议用环境变量）
EMBEDDING_MODEL = "text-embedding-v3"               # 文本嵌入模型：text-embedding-v3（阿里云千问文本嵌入模型）
LLM_MODEL = "qwen-plus"                             # 大语言模型：qwen-plus（阿里云千问-plus模型）

# 初始化chromadb客户端，指定数据库文件路径为./chroma.db
# PersistentClient 会将数据持久化到磁盘，重启后数据不会丢失
chromadb_client = chromadb.PersistentClient("./chroma.db")
# 创建或获取一个名为linghuchong的集合，用于存储向量
# 集合类似于SQL中的表，用于存储一类文档的向量
chromadb_collection = chromadb_client.get_or_create_collection("linghuchong")

# def embed(text: str, store: bool) -> list[float]:#store为True时，将文本转换为向量表示并存储到chromadb中
#     # 调用Gemini模型的embed_content方法，获取文本的向量表示
#     result = google_client.models.embed_content(
#         model=EMBEDDING_MODEL,
#         contents=text,
#         config={
#             "task_type": "RETRIEVAL_DOCUMENT" if store else "RETRIEVAL_QUERY"
#         }
#     )
#     # 检查返回结果是否包含向量表示
#     assert result.embeddings
#     assert result.embeddings[0].values
#     return result.embeddings[0].values


def embed(text: str, store: bool) -> list[float]:
    """
    将文本转换为embedding表示
    Args:
        text: 待嵌入的文本
        store:
            - True: 表示这是文档文本（RETRIEVAL_DOCUMENT），用于建库
            - False: 表示这是查询文本（RETRIEVAL_QUERY），用于检索
    Returns:
        list[float]: 文本的embedding表示（1536维，text-embedding-v3模型）
    """
    task_type = "document" if store else "query"    # document: 文档嵌入，适合长文本建库。query: 查询嵌入，适合短文本检索
    response = TextEmbedding.call(
        model=EMBEDDING_MODEL, input=text, text_type=task_type
    )
    assert response.status_code == HTTPStatus.OK, f"Embedding 失败: {response}"
    return response.output["embeddings"][0]["embedding"]

# 创建数据库
def create_db() -> None:
    """
    1. 获取所有文档 chunks
    2. 对每个 chunk 调用 embed() 获取向量
    3. 使用 upsert() 将 (id, document, embedding) 存入向量数据库
    注意：如果数据库已存在，upsert 会更新已有数据
    """
    for idx, c in enumerate(chunk.get_chunks()):
        print(f"Process: {c}")              # 打印当前处理的chunk
        embedding = embed(c, store=True)
        chromadb_collection.upsert(         # 向集合中插入数据
            ids=[str(idx)], documents=[c], embeddings=[embedding]
        )

# 查询数据库
def query_db(question: str) -> list[str]:
    """
    流程：
    1. 将用户问题转换为向量表示
    2. 在向量数据库中搜索最相似的5个文档
    3. 返回匹配的文档内容
    Args:
        question: 用户的问题字符串
    Returns:
        list[str]: 最相关的5个文档片段列表
    """
    question_embedding = embed(question, store=False)
    result = chromadb_collection.query(query_embeddings=question_embedding, n_results=5)
    assert result["documents"]
    return result["documents"][0]


if __name__ == "__main__":
    """主函数：演示完整的 RAG 流程"""
    # 定义用户问题
    question = "令狐冲领悟了什么魔法？"
    # create_db()  # 已建库就注释掉

    # Step 1: 从向量数据库检索相关文档
    chunks = query_db(question)

    # Step 2: 构建 Prompt
    # 将检索到的文档片段拼接成上下文
    prompt = "请根据以下上下文回答用户问题\n"
    prompt += f"问题: {question}\n"
    prompt += "上下文:\n"
    for c in chunks:
        prompt += f"{c}\n"
        prompt += "-------------\n"

    # Step 3: 调用大语言模型生成回答
    # 使用 OpenAI 兼容接口调用阿里云千问模型
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = client.chat.completions.create(
        model="qwen-plus", messages=[{"role": "user", "content": prompt}]
    )

    # Step 4: 输出最终回答
    print(response.choices[0].message.content)

# Step 3 谷歌Gemini模型生成回答
#     result = google_client.models.generate_content(
#         model=LLM_MODEL,
#         contents=prompt
#     )
#     print(result)