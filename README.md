# AWS MLOps Sentiment Analysis Pipeline

Production-grade sentiment analysis pipeline using AWS services with automated training and deployment.

## Architecture
- Model: DistilBERT for sentiment classification
- Training: AWS SageMaker
- Deployment: Lambda + API Gateway
- Monitoring: CloudWatch
- CI/CD: GitHub Actions

## Setup

### Local Development
```bash
# Install dependencies
poetry install
poetry run pytest

# Configure AWS CLI
sudo apt install awscli
aws configure
```

### AWS Configuration
1. Create IAM User:
   - Name: aws-sentiment-mlops
   - Permissions: 
     - AmazonSageMakerFullAccess
     - AWSLambdaFullAccess
     - AmazonAPIGatewayAdministrator
     - AmazonS3FullAccess
     - CloudWatchFullAccess

2. Configure AWS CLI:
   ```bash
   aws configure
   # Enter Access Key ID
   # Enter Secret Access Key
   # Region: us-east-1
   # Output: json
   ```

3. Verify setup:
   ```bash
   aws sts get-caller-identity
   ```

## Infrastructure
[Architecture diagram coming soon]

## License
MIT

## Status
🚧 Under Development
