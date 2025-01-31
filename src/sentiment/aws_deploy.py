import boto3
import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel

def deploy_model():
    session = sagemaker.Session()
    role = session.get_caller_identity_arn()
    
    # Create HF model
    huggingface_model = HuggingFaceModel(
        model_data=f"s3://{session.default_bucket()}/sentiment-model.tar.gz",
        role=role,
        transformers_version="4.26.0",
        pytorch_version="1.13.1",
        py_version="py39"
    )
    
    # Deploy endpoint
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type="ml.t2.medium"
    )
    
    return predictor

if __name__ == "__main__":
    deploy_model()