import boto3
import os
import json
from dotenv import load_dotenv
load_dotenv()

class RagUtil:

    bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.environ['AWS_REGION'],
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
        )
    namespace = 'user1'
    pinecone_index_name = 'rag-demo-index'
   
    def __init__(self):
      pass

    @classmethod
    def get_embeddings(cls, doc) -> str:
        response = cls.bedrock.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps({"inputText": doc})
            )
        response_body = json.loads(response.get('body').read())
        embedded_text = response_body.get('embedding')
        return embedded_text