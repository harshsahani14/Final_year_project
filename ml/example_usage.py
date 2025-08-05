#!/usr/bin/env python3
"""
Example usage of the fake account detection system
"""

from models.feature_extractor import FeatureExtractor
from models.ml_models import FakeAccountDetectionModels
from models.data_generator import SyntheticDataGenerator
import json

def example_training():
    """Example of training the models"""
    print("=== Training Example ===")
    
    # Generate synthetic data
    data_generator = SyntheticDataGenerator()
    df = data_generator.generate_dataset(n_samples=500, fake_ratio=0.3)
    
    # Initialize components
    feature_extractor = FeatureExtractor()
    ml_models = FakeAccountDetectionModels()
    
    # Extract features
    features_list = []
    labels = []
    
    for _, row in df.iterrows():
        user_data = row.to_dict()
        features = feature_extractor.extract_all_features(user_data)
        features_list.append(features)
        labels.append(user_data['is_fake'])
    
    # Train models
    import pandas as pd
    features_df = pd.DataFrame(features_list)
    X, y = ml_models.prepare_data(features_df, labels)
    results = ml_models.train_models(X, y)
    
    # Print results
    print("Training Results:")
    for model_name, metrics in results.items():
        if 'accuracy' in metrics:
            print(f"  {model_name}: Accuracy={metrics['accuracy']:.4f}")
    
    # Save models
    ml_models.save_models()
    print("Models saved!")

def example_prediction():
    """Example of making predictions"""
    print("\n=== Prediction Example ===")
    
    # Sample user data
    sample_user = {
        "user_id": "test_user_1",
        "username": "suspicious_user123",
        "email": "suspicious_user123@example.com",
        "creation_date": "2024-01-01T00:00:00",
        "follower_count": 10,
        "following_count": 5000,
        "is_verified": False,
        "bio": "Follow me for amazing deals! Click link in bio!",
        "profile_picture_url": "",
        "location": "",
        "website_url": "",
        "posts": [
            {
                "post_id": "post_1",
                "content": "Buy now! Limited time offer!",
                "created_at": "2024-01-01T10:00:00",
                "likes_count": 2,
                "shares_count": 0,
                "comments_count": 0,
                "hashtags": ["#buy", "#deal", "#offer", "#money", "#fast"]
            },
            {
                "post_id": "post_2",
                "content": "Buy now! Limited time offer!",  # Duplicate content
                "created_at": "2024-01-01T10:05:00",  # Posted 5 minutes later
                "likes_count": 1,
                "shares_count": 0,
                "comments_count": 0,
                "hashtags": ["#buy", "#deal", "#offer", "#money", "#fast"]
            }
        ]
    }
    
    # Initialize components
    feature_extractor = FeatureExtractor()
    ml_models = FakeAccountDetectionModels()
    
    # Try to load existing models
    try:
        ml_models.load_models()
        if not ml_models.trained_models:
            print("No trained models found. Training first...")
            example_training()
            ml_models.load_models()
    except:
        print("Training models first...")
        example_training()
        ml_models.load_models()
    
    # Extract features
    features = feature_extractor.extract_all_features(sample_user)
    
    # Make prediction
    prediction = ml_models.ensemble_predict(features)
    
    # Print results
    print(f"User: {sample_user['username']}")
    print(f"Risk Level: {prediction['risk_level']}")
    print(f"Fake Probability: {prediction['ensemble_probability']:.4f}")
    print(f"Confidence: {prediction['ensemble_confidence']:.4f}")
    
    print("\nIndividual Model Predictions:")
    for model_name, pred in prediction['individual_predictions'].items():
        print(f"  {model_name}: {pred['probability']:.4f}")
    
    print("\nTop Features:")
    for feature, value in list(features.items())[:10]:
        print(f"  {feature}: {value:.4f}")

if __name__ == "__main__":
    # Run examples
    example_training()
    example_prediction()
