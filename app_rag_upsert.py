import os
import boto3
import json

from app_rag_data import docs_data
from app_rag_utils import RagUtil
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()


pinecone_index_name = RagUtil.pinecone_index_name
# Create Index if not exists
pc = Pinecone(os.environ['PINECONE_API_KEY'])
if pc.has_index(pinecone_index_name):
    print(f'Index {pinecone_index_name} already exists.')
else:
    pc.create_index_for_model(
        name=pinecone_index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )
    print(f'Index {pinecone_index_name} created')

# Initialize boto3 client for Amazon Bedrock
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.environ['AWS_REGION'],
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
)

#Upsert documents to Pinecon
index = pc.Index(pinecone_index_name)
for doc in docs_data:
    exists =  index.fetch(ids=[doc['id']], namespace=RagUtil.namespace)
    if doc["id"] not in exists.vectors:
        
        # Get embedding of the doc
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({"inputText": doc['text']})
        )
        response_body = json.loads(response.get('body').read())
        embedded_text = response_body.get('embedding')

        # Upsert embeddings
        vector = {
            'id': doc["id"],
            'values': embedded_text,
            'metadata': {**doc["metadata"], 'text': doc["text"]}
        }
        index.upsert(vectors=[vector],namespace=docs_namespace)
        print('Vector upserted successfully')
    else:
        print('Document already exists')