# tests/test_lambda.py
import json
import pytest
from lambda.app import lambda_handler

def test_lambda_handler():
    event = {
        'body': json.dumps({
            'text': 'This is a great test!'
        })
    }
    response = lambda_handler(event, {})
    assert response['statusCode'] == 200
    body = json.loads(response['body'])
    assert 'sentiment' in body
    assert 'score' in body

def test_lambda_bad_input():
    event = {
        'body': 'invalid json'
    }
    with pytest.raises(json.JSONDecodeError):
        lambda_handler(event, {})