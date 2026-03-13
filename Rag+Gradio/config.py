import os
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv()

# 获取 OpenAI API 密钥
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# 官方文档 - Models：https://platform.openai.com/docs/models
MODELS = [
    "gpt-3.5-turbo-1106", 
    "gpt-3.5-turbo", 
    "gpt-3.5-turbo-16k", 
    "gpt-4-1106-preview", 
    "gpt-4",  
    'qwen-plus',
]
DEFAULT_MODEL = MODELS[-1]
MODEL_TO_MAX_TOKENS = {
    "gpt-3.5-turbo-1106": 4096,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-4-1106-preview": 4096,
    "gpt-4": 8192,
}

#当前用localhost，Qdrant和Gradio运行在同一台机器上。
#如果要把Qdrant部署到服务器上，就改成服务器的内网IP。
QDRANT_HOST = "localhost"
# QDRANT_HOST = "192.168.247.128"
QDRANT_PORT = 6333

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

DEFAULT_MAX_TOKENS = 2000