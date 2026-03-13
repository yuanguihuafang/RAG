# RAG 系统

基于检索增强生成（RAG）的中文文档问答系统。

## 系统架构

RAG系统主要由以下六个模块组成：

### 1. 文档分片 (Chunk)

将原始文档分割成较小的文本块，便于后续处理和检索。

```python
def split_into_chunks(doc_file: str) -> List[str]:
    with open(doc_file, 'r') as file:
        content = file.read()
    return [chunk for chunk in content.split("\n\n")]
```

### 2. 文本 Embedding 生成

使用中文预训练模型将文本块转换为向量表示。

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")

def embed_chunk(chunk: str) -> List[float]:
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()
```

### 3. 向量数据库

使用 ChromaDB 存储文本块及其对应的嵌入向量。

```python
import chromadb

chromadb_client = chromadb.EphemeralClient()
chromadb_collection = chromadb_client.get_or_create_collection(name="default")

def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chromadb_collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)]
        )
```

### 4. 召回 (Retrieval)

根据用户查询从向量数据库中检索最相关的文本块。

```python
def retrieve(query: str, top_k: int) -> List[str]:
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]
```

### 5. 重新排序 (Rerank)

使用交叉编码器对检索结果进行二次排序，提高相关性。

```python
from sentence_transformers import CrossEncoder

def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)
    
    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    return [chunk for chunk, _ in scored_chunks][:top_k]
```

### 6. 生成回答 (Generation)

使用大语言模型基于检索到的内容生成最终回答。

```python
from google import genai

def generate(query: str, chunks: List[str]) -> str:
    prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。

用户问题: {query}

相关片段:
{"\n\n".join(chunks)}

请基于上述内容作答，不要编造信息。"""

    response = google_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text
```

## 技术栈

- **向量嵌入**: sentence-transformers (shibing624/text2vec-base-chinese)
- **向量数据库**: ChromaDB
- **重排序模型**: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
- **大语言模型**: Google Gemini 2.5 Flash

pyproject.toml 是 uv (现代Python包管理工具) 的项目配置文件，用于定义项目元数据和依赖。  
作用：声明项目名称、版本、描述；指定 Python 版本要求 (>=3.12)；列出依赖包及版本范围。  
与 requirements.txt 的关系：requirements.txt 是传统方式，手动列出依赖；  
pyproject.toml 是 PEP 621 标准，uv 会自动从中读取依赖；使用 uv add xxx 时会自动更新此文件。  

uv.lock 是锁定文件，记录项目所有依赖的精确版本。  


## 环境配置

### 安装依赖

首先请确保你的系统已经安装了 uv 和 Jupyter，否则请参照如下链接安装：

- uv: https://docs.astral.sh/uv/getting-started/installation/
- Jupyter: https://jupyter.org/install

然后在项目根目录下创建一个名为 `.env` 的文件，并添加以下内容：

```env
GEMINI_API_KEY=xxx
```

其中 xxx 为你的 Google Gemini API 密钥。没有密钥的用户可以在 https://aistudio.google.com/apikey 上申请。

然后使用 uv 安装如下 Python 依赖：

```bash
uv add sentence_transformers chromadb google-genai python-dotenv
```

### 运行

使用 uv 运行 Jupyter Notebook：

```bash
uv run --with jupyter jupyter lab
```
