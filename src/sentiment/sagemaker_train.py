import sagemaker
from sagemaker.huggingface import HuggingFace

def create_training_job():
    sagemaker_session = sagemaker.Session()
    role = sagemaker_session.get_caller_identity_arn()

    huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='src/sentiment',
        instance_type='ml.m5.xlarge',
        instance_count=1,
        role=role,
        transformers_version='4.26.0',
        pytorch_version='1.13.1',
        py_version='py39',
    )
    
    return huggingface_estimator