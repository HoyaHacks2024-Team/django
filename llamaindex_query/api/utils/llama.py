import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Rest of your imports
import json
import openai
from llama_index.llms import AzureOpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

# Your existing code for setting up AzureOpenAI and OpenAIEmbedding
# Set up the AzureOpenAI instance
llm = AzureOpenAI(
    model_name=os.getenv("OPENAI_MODEL_COMPLETION"),
    deployment_name=os.getenv("OPENAI_MODEL_COMPLETION"),
    api_base=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_type=os.getenv("OPENAI_API_TYPE"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=0,
)

# Set up the OpenAIEmbedding instance
embed_model = OpenAIEmbedding(
    model=os.getenv("OPENAI_MODEL_EMBEDDING"),
    deployment_name=os.getenv("OPENAI_DEPLOYMENT_EMBEDDING"),
    api_base=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_type=os.getenv("OPENAI_API_TYPE"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)
