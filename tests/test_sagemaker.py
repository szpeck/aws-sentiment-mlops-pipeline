# tests/test_sagemaker.py
import pytest
from sagemaker.huggingface import HuggingFace

def test_training_job():
    try:
        estimator = HuggingFace(
            entry_point='train.py',
            source_dir='src/sentiment',
            instance_type='ml.m5.xlarge',
            instance_count=1,
            role='arn:aws:iam::role/test',
            transformers_version='4.26.0',
            pytorch_version='1.13.1',
            py_version='py39',
        )
        assert estimator is not None
    except Exception as e:
        pytest.fail(f"Failed to create estimator: {str(e)}")