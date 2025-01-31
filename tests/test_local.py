# tests/test_local.py
from src.sentiment.inference import SentimentPredictor

def test_local_prediction():
   predictor = SentimentPredictor()
   result = predictor.predict("This is a great test!")
   assert isinstance(result, dict)
   assert 'sentiment' in result
   assert 'score' in result