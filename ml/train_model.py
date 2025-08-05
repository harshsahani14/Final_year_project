#!/usr/bin/env python3
"""
Training script for fake social media account detection models
"""

import pandas as pd
import numpy as np
import logging
from models.feature_extractor import FeatureExtractor
from models.ml_models import FakeAccountDetectionModels
from models.data_generator import SyntheticDataGenerator
import argparse
import json
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train fake account detection models')
    parser.add_argument('--data-file', type=str, help='Path to training data file')
    parser.add_argument('--generate-data', action='store_true', help='Generate synthetic data')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--fake-ratio', type=float, default=0.3, help='Ratio of fake accounts')
    parser.add_argument('--model-dir', type=str, default='saved_models', help='Directory to save models')
    parser.add_argument('--tune-hyperparams', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--output-report', type=str, default='training_report.json', help='Output report file')
    
    args = parser.parse_args()
    
    # Initialize components
    feature_extractor = FeatureExtractor()
    ml_models = FakeAccountDetectionModels(model_dir=args.model_dir)
    
    # Load or generate data
    if args.generate_data or not args.data_file:
        logger.info("Generating synthetic data...")
        data_generator = SyntheticDataGenerator()
        df = data_generator.generate_dataset(n_samples=args.n_samples, fake_ratio=args.fake_ratio)
        
        # Save generated data
        data_file = 'synthetic_data.json'
        data_generator.save_dataset(df, data_file)
        logger.info(f"Generated {len(df)} samples and saved to {data_file}")
    else:
        logger.info(f"Loading data from {args.data_file}")
        data_generator = SyntheticDataGenerator()
        df = data_generator.load_dataset(args.data_file)
    
    # Extract features
    logger.info("Extracting features...")
    features_list = []
    labels = []
    
    for idx, row in df.iterrows():
        try:
            user_data = row.to_dict()
            features = feature_extractor.extract_all_features(user_data)
            features_list.append(features)
            labels.append(user_data['is_fake'])
        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
            continue
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    logger.info(f"Extracted features for {len(features_df)} samples")
    logger.info(f"Feature columns: {list(features_df.columns)}")
    
    # Prepare data for training
    X, y = ml_models.prepare_data(features_df, labels)
    
    # Hyperparameter tuning (optional)
    if args.tune_hyperparams:
        logger.info("Performing hyperparameter tuning...")
        best_models = ml_models.hyperparameter_tuning(X, y)
        logger.info("Hyperparameter tuning completed")
    
    # Train models
    logger.info("Training models...")
    training_results = ml_models.train_models(X, y)
    
    # Save models
    ml_models.save_models()
    logger.info(f"Models saved to {args.model_dir}")
    
    # Generate feature importance reports
    feature_importance_reports = {}
    for model_name in ['random_forest']:  # Only tree-based models
        try:
            importance = ml_models.get_feature_importance(model_name)
            feature_importance_reports[model_name] = importance
            
            # Log top 10 features
            logger.info(f"\nTop 10 features for {model_name}:")
            for feature, importance_score in list(importance.items())[:10]:
                logger.info(f"  {feature}: {importance_score:.4f}")
                
        except Exception as e:
            logger.error(f"Error getting feature importance for {model_name}: {str(e)}")
    
    # Prepare final report
    report = {
        'training_info': {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'fake_ratio': sum(y) / len(y),
            'feature_names': ml_models.feature_names
        },
        'model_performance': training_results,
        'feature_importance': feature_importance_reports
    }
    
    # Save report
    with open(args.output_report, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Training completed! Report saved to {args.output_report}")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Fake accounts: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print("\nModel Performance:")
    for model_name, results in training_results.items():
        if 'accuracy' in results:
            print(f"  {model_name}: Accuracy={results['accuracy']:.4f}, AUC={results['auc_score']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
