import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import logging
from typing import Dict, List, Tuple, Any
import os

class FakeAccountDetectionModels:
    """ML Models for fake social media account detection"""
    
    def __init__(self, model_dir: str = "saved_models"):
        self.model_dir = model_dir
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        self.trained_models = {}
        self.feature_names = []
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_data(self, features_df: pd.DataFrame, labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Store feature names
        self.feature_names = list(features_df.columns)
        
        # Convert to numpy arrays
        X = features_df.values
        y = np.array(labels)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        self.logger.info(f"Data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        
        return X_scaled, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Dict]:
        """Train all models and return performance metrics"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        results = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = model.score(X_test, y_test)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                # Store trained model
                self.trained_models[model_name] = model
                
                # Store results
                results[model_name] = {
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                self.logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform hyperparameter tuning for models"""
        self.logger.info("Starting hyperparameter tuning...")
        
        # Parameter grids for each model
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly']
            },
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        best_models = {}
        
        for model_name, model in self.models.items():
            if model_name in param_grids:
                self.logger.info(f"Tuning {model_name}...")
                
                try:
                    grid_search = GridSearchCV(
                        model, 
                        param_grids[model_name],
                        cv=5,
                        scoring='roc_auc',
                        n_jobs=-1,
                        verbose=1
                    )
                    
                    grid_search.fit(X, y)
                    
                    best_models[model_name] = {
                        'best_model': grid_search.best_estimator_,
                        'best_params': grid_search.best_params_,
                        'best_score': grid_search.best_score_
                    }
                    
                    # Update the model with best parameters
                    self.models[model_name] = grid_search.best_estimator_
                    
                    self.logger.info(f"{model_name} best score: {grid_search.best_score_:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error tuning {model_name}: {str(e)}")
        
        return best_models
    
    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importance for tree-based models"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return dict(sorted_features)
        else:
            self.logger.warning(f"Model {model_name} doesn't support feature importance")
            return {}
    
    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict for a single user"""
        if not self.trained_models:
            raise ValueError("No trained models available")
        
        # Convert features to array
        feature_array = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
        
        # Scale features
        feature_array_scaled = self.scaler.transform(feature_array)
        
        predictions = {}
        
        for model_name, model in self.trained_models.items():
            try:
                # Get prediction probability
                prob = model.predict_proba(feature_array_scaled)[0, 1]
                prediction = model.predict(feature_array_scaled)[0]
                
                predictions[model_name] = {
                    'probability': float(prob),
                    'prediction': int(prediction),
                    'confidence': float(max(model.predict_proba(feature_array_scaled)[0]))
                }
                
            except Exception as e:
                self.logger.error(f"Error predicting with {model_name}: {str(e)}")
        
        return predictions
    
    def ensemble_predict(self, features: Dict[str, float], weights: Dict[str, float] = None) -> Dict[str, Any]:
        """Make ensemble prediction using multiple models"""
        individual_predictions = self.predict_single(features)
        
        if not individual_predictions:
            return {'error': 'No predictions available'}
        
        # Default equal weights
        if weights is None:
            weights = {name: 1.0 for name in individual_predictions.keys()}
        
        # Calculate weighted average
        total_weight = sum(weights.values())
        weighted_prob = 0.0
        weighted_confidence = 0.0
        
        for model_name, pred in individual_predictions.items():
            weight = weights.get(model_name, 1.0) / total_weight
            weighted_prob += pred['probability'] * weight
            weighted_confidence += pred['confidence'] * weight
        
        # Final prediction
        final_prediction = 1 if weighted_prob > 0.5 else 0
        
        return {
            'ensemble_probability': weighted_prob,
            'ensemble_prediction': final_prediction,
            'ensemble_confidence': weighted_confidence,
            'individual_predictions': individual_predictions,
            'risk_level': self._determine_risk_level(weighted_prob)
        }
    
    def _determine_risk_level(self, probability: float) -> str:
        """Determine risk level based on probability"""
        if probability < 0.3:
            return 'LOW'
        elif probability < 0.6:
            return 'MEDIUM'
        elif probability < 0.8:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def save_models(self) -> None:
        """Save all trained models to disk"""
        if not self.trained_models:
            self.logger.warning("No trained models to save")
            return
        
        # Save scaler
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature names
        features_path = os.path.join(self.model_dir, 'feature_names.pkl')
        joblib.dump(self.feature_names, features_path)
        
        # Save each model
        for model_name, model in self.trained_models.items():
            model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
            joblib.dump(model, model_path)
            self.logger.info(f"Saved {model_name} to {model_path}")
    
    def load_models(self) -> None:
        """Load trained models from disk"""
        try:
            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            # Load feature names
            features_path = os.path.join(self.model_dir, 'feature_names.pkl')
            if os.path.exists(features_path):
                self.feature_names = joblib.load(features_path)
            
            # Load models
            for model_name in self.models.keys():
                model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
                if os.path.exists(model_path):
                    self.trained_models[model_name] = joblib.load(model_path)
                    self.logger.info(f"Loaded {model_name} from {model_path}")
            
            self.logger.info(f"Loaded {len(self.trained_models)} models")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
    
    def evaluate_model_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """Evaluate performance of all trained models"""
        if not self.trained_models:
            raise ValueError("No trained models available")
        
        results = {}
        
        for model_name, model in self.trained_models.items():
            try:
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Metrics
                accuracy = model.score(X_test, y_test)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
