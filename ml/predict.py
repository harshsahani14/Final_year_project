#!/usr/bin/env python3
"""
Prediction script for fake social media account detection
"""

import json
import argparse
import logging
from models.feature_extractor import FeatureExtractor
from models.ml_models import FakeAccountDetectionModels

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Predict fake social media accounts')
    parser.add_argument('--user-data', type=str, required=True, help='Path to user data JSON file')
    parser.add_argument('--model-dir', type=str, default='saved_models', help='Directory with saved models')
    parser.add_argument('--output', type=str, help='Output file for predictions')
    
    args = parser.parse_args()
    
    # Initialize components
    feature_extractor = FeatureExtractor()
    ml_models = FakeAccountDetectionModels(model_dir=args.model_dir)
    
    # Load models
    logger.info("Loading trained models...")
    ml_models.load_models()
    
    if not ml_models.trained_models:
        logger.error("No trained models found. Please train models first.")
        return
    
    # Load user data
    logger.info(f"Loading user data from {args.user_data}")
    with open(args.user_data, 'r') as f:
        user_data = json.load(f)
    
    # Handle single user or list of users
    if isinstance(user_data, dict):
        users = [user_data]
    else:
        users = user_data
    
    # Make predictions
    predictions = []
    
    for i, user in enumerate(users):
        try:
            logger.info(f"Processing user {i+1}/{len(users)}: {user.get('username', 'unknown')}")
            
            # Extract features
            features = feature_extractor.extract_all_features(user)
            
            # Make ensemble prediction
            prediction = ml_models.ensemble_predict(features)
            
            # Add user info to prediction
            prediction['user_id'] = user.get('user_id', f'user_{i}')
            prediction['username'] = user.get('username', 'unknown')
            
            predictions.append(prediction)
            
            # Log result
            risk_level = prediction['risk_level']
            probability = prediction['ensemble_probability']
            logger.info(f"  Result: {risk_level} risk (probability: {probability:.4f})")
            
        except Exception as e:
            logger.error(f"Error processing user {i}: {str(e)}")
            predictions.append({
                'user_id': user.get('user_id', f'user_{i}'),
                'username': user.get('username', 'unknown'),
                'error': str(e)
            })
    
    # Save predictions
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        logger.info(f"Predictions saved to {args.output}")
    else:
        # Print predictions
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        
        for pred in predictions:
            if 'error' in pred:
                print(f"User: {pred['username']} - ERROR: {pred['error']}")
            else:
                print(f"User: {pred['username']}")
                print(f"  Risk Level: {pred['risk_level']}")
                print(f"  Probability: {pred['ensemble_probability']:.4f}")
                print(f"  Confidence: {pred['ensemble_confidence']:.4f}")
                print()

if __name__ == "__main__":
    main()
