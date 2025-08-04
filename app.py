from pinecone import Pinecone
import boto3
import json
from dotenv import load_dotenv
load_dotenv()
import os

# Initialize Pinecone
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

index_name = "developer-quickstart-py"

if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )
print('index created or skipped')

# Initialize boto3 client for Amazon Bedrock
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.environ['AWS_REGION'],
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
)

def get_embedding(text):
    """Get embeddings using Amazon Titan embedding model"""
    try:
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({"inputText": text})
        )
        response_body = json.loads(response.get('body').read())
        return response_body.get('embedding')
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return None

def invoke_llm(prompt):
    """Invoke Bedrock LLM using Titan model"""
    try:
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0,
                "topP": 1,
            }
        })
        
        response = bedrock.invoke_model(
            modelId="amazon.titan-text-express-v1",
            body=body
        )
        response_body = json.loads(response.get('body').read())
        return response_body.get('results')[0].get('outputText')
    except Exception as e:
        print(f"Error invoking LLM: {str(e)}")
        return None

import time

# Wait for index to be ready
time.sleep(1)

# Define the documents with their IDs
docs_data = [
    {
        "id": "pradip_salary_2024",
        "text": "Pradip's total salary for the year 2024 is 3000000",
        "metadata": {"source": "salary"}
    },
    {
        "id": "pradip_loans",
        "text": "Pradip's total loans amounts 1000000",
        "metadata": {"source": "loans"}
    },
    {
        "id": "pradip_tax",
        "text": "His tax bracket is 25%",
        "metadata": {"source": "tax"}
    }
]

# Get the index instance
index = pc.Index(index_name)

# Check and add documents one by one
for doc in docs_data:
    try:
        # Check if document exists
        existing_doc = index.fetch(
            ids=[doc["id"]],
            namespace="employee_data"
        )
        
        # Check if the document exists using the vectors property of FetchResponse
        if not hasattr(existing_doc, 'vectors') or doc["id"] not in existing_doc.vectors:
            # Get embeddings for the text
            embedded_text = get_embedding(doc["text"])
            
            if embedded_text:
                # Upsert to Pinecone
                index.upsert(
                    vectors=[{
                        'id': doc["id"],
                        'values': embedded_text,
                        'metadata': {
                            **doc["metadata"],
                            'text': doc["text"]  # Store the original text in metadata
                        }
                    }],
                    namespace="employee_data"
                )
                print(f"Added document with ID: {doc['id']}")
            else:
                print(f"Failed to get embedding for document {doc['id']}")
        else:
            print(f"Document with ID {doc['id']} already exists, skipping.")
    except Exception as e:
        print(f"Error checking/adding document {doc['id']}: {str(e)}")

def generate_response(query):
    # Get embeddings for the query
    query_embedding = get_embedding(query)
    
    if not query_embedding:
        return "Failed to process query"
    
    # Search in Pinecone using the index
    search_results = index.query(
        vector=query_embedding,
        top_k=1,
        namespace="employee_data",
        include_metadata=True
    )
    
    # Extract the content from the most relevant document
    if hasattr(search_results, 'matches') and search_results.matches:
        context = search_results.matches[0].metadata['text']
        prompt = f"""Based on the following information, please answer the question.
        
Information: {context}

Question: {query}

Please extract and state the exact information from the provided context to answer the question."""
        
        response = invoke_llm(prompt)
        return response if response else "Failed to generate response"
    return "No relevant information found."

# Example usage
query = "Pradip's total loans amounts?"
response = generate_response(query)
print(f"Query: {query}")
print(f"Response: {response}")

