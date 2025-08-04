from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_aws import BedrockLLM
from langchain_aws.embeddings import BedrockEmbeddings

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

llm = BedrockLLM(
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    region_name=os.environ['AWS_REGION'],
    model_id="amazon.titan-text-express-v1"
)

# res = llm.invoke(input="What is the recipe of mayonnaise?")
# print(res)

import time

# Initialize the Bedrock embeddings model
embeddings = BedrockEmbeddings(
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    region_name=os.environ['AWS_REGION'],
    model_id="amazon.titan-embed-text-v2:0"
)

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
            embedded_text = embeddings.embed_query(doc["text"])
            
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
            print(f"Document with ID {doc['id']} already exists, skipping.")
    except Exception as e:
        print(f"Error checking/adding document {doc['id']}: {str(e)}")


from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate(
    template="""Based on the following information, please answer the question.
    Information: {context}
    Question: {query}
    Please extract and state the exact information from the provided context to answer the question.
    """,
        input_variables=["context", "query"]
    )
def generate_response(query):
    # Get embeddings for the query
    query_embedding = embeddings.embed_query(query)
    
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
        prompt_text = prompt.format(context=context, query=query)
        response = llm.invoke(input=prompt_text)
        return response
    return "No relevant information found."

# Example usage
query = "What is Pradip's total loans amounts?"
response = generate_response(query)
print(f"Query: {query}")
print(f"Response: {response}")

