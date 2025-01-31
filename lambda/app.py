import json
import boto3
import os

runtime = boto3.client('runtime.sagemaker')
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']

def lambda_handler(event, context):
    body = json.loads(event['body'])
    text = body['text']
    
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=json.dumps({'inputs': text})
    )
    
    result = json.loads(response['Body'].read().decode())
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'sentiment': result[0]['label'],
            'score': result[0]['score']
        })
    }