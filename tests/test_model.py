# tests/test_model.py
import pytest
from src.sentiment.inference import SentimentPredictor

def test_predictor_init():
   predictor = SentimentPredictor()
   assert predictor.pipeline is not None

def test_positive_sentiment():
   predictor = SentimentPredictor()
   result = predictor.predict("This is fantastic!")
   assert isinstance(result, dict)
   assert "sentiment" in result
   assert "confidence" in result
   assert isinstance(result["confidence"], float)

def test_negative_sentiment():
   predictor = SentimentPredictor()
   result = predictor.predict("This is terrible!")
   assert isinstance(result, dict)
   assert "sentiment" in result
   assert result["confidence"] > 0.5