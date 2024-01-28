"""Azure CosmosDB MongoDB Core Demo

# Azure CosmosDB MongoDB Vector Store
In this notebook we are going to show how to use Azure Cosmosdb Mongodb vCore to perform vector searches in LlamaIndex. We will create the embedding using Azure Open AI.
"""

from llama_index.llms import AzureOpenAI
from llama_index.embeddings import VoyageEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.storage.storage_context import StorageContext
from llama_index import set_global_service_context
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

import logging
import sys
import os

import pymongo

from dotenv import load_dotenv
load_dotenv()

"""### Setup Azure OpenAI
The first step is to configure the models. They will be used to create embeddings for the documents loaded into the db and for llm completions.
"""
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Set up the AzureOpenAI instance
llm = AzureOpenAI(
    model=os.getenv("OPENAI_MODEL_COMPLETION"),
    deployment_name=os.getenv("OPENAI_DEPLOYMENT_COMPLETION"),
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

# Set up the OpenAIEmbedding instance
embed_model = VoyageEmbedding(model_name=os.getenv("VOYAGE_MODEL_NAME"), voyage_api_key=os.getenv("VOYAGE_API_KEY"))

# Set up service context
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
set_global_service_context(service_context)

"""### Create the index
Here we establish the connection to an Azure Cosmosdb mongodb vCore cluster and create an vector search index.
"""

mongodb_client = pymongo.MongoClient(os.getenv("CONNECTION_STRING"))

db_name=os.getenv("DB_NAME")
collection_name="paul_graham_essay"

store = MongoDBAtlasVectorSearch(
    mongodb_client=mongodb_client,
    db_name=db_name,
    collection_name=collection_name,
)

db = mongodb_client.get_database(db_name)
collection = db.get_collection(collection_name)

# Define a college and query
college = "Dickinson College"
query = "What did the author love working on?"

storage_context = StorageContext.from_defaults(
    vector_store=store
)

index = VectorStoreIndex.from_documents(store)

# Perform the query
query_engine = store.as_query_engine()
answer = query_engine.query(query)

# Print the results
print(answer.get_formatted_sources())
print("Query was:", query)
print("Answer was:", answer)