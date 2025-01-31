# src/sentiment/inference.py
from transformers import pipeline

class SentimentPredictor:
   def __init__(self, model_path=None):
       if model_path:
           self.pipeline = pipeline("sentiment-analysis", model=model_path)
       else:
           self.pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
   
   def predict(self, text):
       result = self.pipeline(text)[0]
       return {
           "sentiment": result["label"],
           "confidence": float(result["score"])
       }

# Example usage
if __name__ == "__main__":
   predictor = SentimentPredictor()
   result = predictor.predict("This movie was amazing!")
   print(result)