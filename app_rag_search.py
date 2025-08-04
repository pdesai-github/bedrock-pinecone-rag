from app_rag_utils import RagUtil
from pinecone import Pinecone
import os
import boto3
import json
query = "Tell me about Pradip's financial data?"

# Search relevant context in Pinecone
# Get embeddings for query
embeddings = RagUtil.get_embeddings(query)
pc = Pinecone(os.environ['PINECONE_API_KEY'])
index = pc.Index(RagUtil.pinecone_index_name)
search_results = index.query(
        vector=embeddings,
        top_k=5,
        namespace=RagUtil.namespace,
        include_metadata=True
    )

if hasattr(search_results, 'matches') and search_results.matches:
    context = "\n".join([match.metadata['text'] for match in search_results.matches])
    print(f"context retrieved - {context}")

    prompt_template = f"""Based on the following information, please answer the question.                
        Information: {context}
        Question: {query}
        Please extract and state the exact information from the provided context to answer the question in a clear and concise manner."""
    
    # Invoke LLM
    body = json.dumps({
            "inputText": prompt_template,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0,
                "topP": 1,
            }
        })
    
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=os.environ['AWS_REGION'],
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
    )
    response = bedrock.invoke_model(
        modelId="amazon.titan-text-express-v1",
        body=body
    )
    response_body = json.loads(response.get('body').read())
    output = response_body.get('results')[0].get('outputText')
    print(f"Output - {output}")
else:
    print('No relevant data found')
