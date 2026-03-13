# RAG2 检索增强生成系统

基于检索增强生成（RAG）的中文文档问答系统，使用阿里云千问 API 实现文本嵌入与问答生成。

## 系统架构

RAG2 系统主要由以下四个模块组成：

### 1. 文档分片 (Chunk)
将原始文档按章节分割成较小的文本块，保留章节标题信息以便保持上下文完整性。  

### 2. 文本 Embedding 生成
使用阿里云千问文本嵌入模型将文本块转换为向量表示。

### 3. 向量数据库
使用 ChromaDB 持久化存储文本块及其对应的嵌入向量。

### 4. 召回与生成

根据用户查询从向量数据库中检索最相关的文本块，然后调用大语言模型生成回答。

## 技术栈

- **文本嵌入**: 阿里云千问 text-embedding-v3
- **向量数据库**: ChromaDB (持久化)
- **大语言模型**: 阿里云千问 qwen-plus
- **API 兼容**: OpenAI 兼容协议

## 环境配置

### 安装依赖

```bash
pip install chromadb dashscope openai python-dotenv
```

### 配置环境变量

在项目根目录下创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=你的千问api密钥
```

获取 API Key：https://dashscope.console.aliyun.com/

### 运行

```bash
python embed.py
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `chunk.py` | 文档分片模块 |
| `embed.py` | 嵌入生成、向量存储、检索问答模块 |
| `data.md` | 文档数据（令狐冲转生为史莱姆的故事） |
| `.env` | 环境变量配置文件 |
