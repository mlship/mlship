#!/usr/bin/env python3

import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import json
from pathlib import Path
import logging
import torch
import torch.nn as nn
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from PIL import Image
import io
from mlship.models.wrapper import ModelWrapper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_numeric_regression_model():
    """Create a simple numeric regression model"""
    try:
        logger.info("Generating data for regression model...")
        X = np.random.rand(100, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
        
        logger.info("Training regression model...")
        model = LinearRegression()
        model.fit(X, y)
        
        # Wrap the model
        wrapped_model = ModelWrapper(
            model=model,
            model_type="NumericRegression",
            input_type="numeric",
            output_type="numeric",
            feature_names=['x1', 'x2']
        )
        
        logger.info("Saving regression model...")
        save_path = Path('models/numeric/regression.joblib')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(wrapped_model, save_path)
        logger.info(f"Created regression model at {save_path}")
        
        # Test the model
        test_model = joblib.load(save_path)
        test_pred = test_model.predict([[0.5, 0.5]])
        logger.info(f"Test prediction: {test_pred}")
        
    except Exception as e:
        logger.error(f"Error creating regression model: {str(e)}", exc_info=True)
        raise

def create_numeric_classification_model():
    """Create a simple numeric classification model"""
    try:
        logger.info("Generating data for classification model...")
        X = np.random.rand(100, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        
        logger.info("Training classification model...")
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        
        # Wrap the model
        wrapped_model = ModelWrapper(
            model=model,
            model_type="NumericClassification",
            input_type="numeric",
            output_type="label",
            feature_names=['x1', 'x2'],
            classes=[0, 1]
        )
        
        logger.info("Saving classification model...")
        save_path = Path('models/numeric/classification.joblib')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(wrapped_model, save_path)
        logger.info(f"Created classification model at {save_path}")
        
        # Test the model
        test_model = joblib.load(save_path)
        test_pred = test_model.predict([[0.5, 0.5]])
        logger.info(f"Test prediction: {test_pred}")
        
    except Exception as e:
        logger.error(f"Error creating classification model: {str(e)}", exc_info=True)
        raise

def create_text_sentiment_model():
    """Create a text sentiment analysis model"""
    try:
        logger.info("Loading text sentiment model...")
        sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # Wrap the model
        wrapped_model = ModelWrapper(
            model=sentiment,
            model_type="TextSentiment",
            input_type="text",
            output_type="label",
            feature_names=["text"]
        )
        
        logger.info("Saving sentiment model...")
        save_path = Path('models/text/sentiment.joblib')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(wrapped_model, save_path)
        logger.info(f"Created sentiment model at {save_path}")
        
        # Test the model
        test_model = joblib.load(save_path)
        test_pred = test_model.predict(["This is a great day!"])
        logger.info(f"Test prediction: {test_pred}")
        
    except Exception as e:
        logger.error(f"Error creating sentiment model: {str(e)}", exc_info=True)
        raise

def create_text_generation_model():
    """Create a text generation model"""
    try:
        logger.info("Loading text generation model...")
        generator = pipeline("text-generation", model="gpt2")
        
        # Wrap the model
        wrapped_model = ModelWrapper(
            model=generator,
            model_type="TextGeneration",
            input_type="text",
            output_type="text",
            feature_names=["prompt"]
        )
        
        logger.info("Saving generation model...")
        save_path = Path('models/text/generator.joblib')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(wrapped_model, save_path)
        logger.info(f"Created generation model at {save_path}")
        
        # Test the model
        test_model = joblib.load(save_path)
        test_pred = test_model.predict(["Once upon a time"])
        logger.info(f"Test prediction: {test_pred}")
        
    except Exception as e:
        logger.error(f"Error creating generation model: {str(e)}", exc_info=True)
        raise

def create_image_classification_model():
    """Create an image classification model"""
    try:
        logger.info("Loading image classification model...")
        classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
        
        # Wrap the model
        wrapped_model = ModelWrapper(
            model=classifier,
            model_type="ImageClassification",
            input_type="image",
            output_type="label",
            feature_names=["image"]
        )
        
        logger.info("Saving classification model...")
        save_path = Path('models/image/classifier.joblib')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(wrapped_model, save_path)
        logger.info(f"Created image classification model at {save_path}")
        
    except Exception as e:
        logger.error(f"Error creating image classification model: {str(e)}", exc_info=True)
        raise

def main():
    """Create all test models"""
    try:
        logger.info("Starting test model creation...")
        current_dir = Path.cwd()
        logger.info(f"Current working directory: {current_dir}")
        
        models_dir = current_dir / 'models'
        logger.info(f"Models will be saved in: {models_dir}")
        
        # Create models
        logger.info("\nCreating numeric regression model...")
        create_numeric_regression_model()
        
        logger.info("\nCreating numeric classification model...")
        create_numeric_classification_model()
        
        logger.info("\nCreating text sentiment model...")
        create_text_sentiment_model()
        
        logger.info("\nCreating text generation model...")
        create_text_generation_model()
        
        logger.info("\nCreating image classification model...")
        create_image_classification_model()
        
        logger.info("\nAll models created successfully!")
        
        # List created models
        logger.info("\nCreated models:")
        for model_path in Path('models').rglob('*.joblib'):
            model = joblib.load(model_path)
            info = model.get_model_info()
            logger.info(f"\n- {model_path}:")
            logger.info(f"  Type: {info['type']}")
            logger.info(f"  Input: {info['input_type']}")
            logger.info(f"  Output: {info['output_type']}")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 