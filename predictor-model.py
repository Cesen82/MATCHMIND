"""
Advanced Over 2.5 Goals Prediction Model with Ensemble Learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import joblib
from datetime import datetime
import logging

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from sklearn.calibration import CalibratedClassifierCV

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available, using fallback models")

from config import MODEL_CONFIG, FEATURE_CONFIG, DATA_DIR
from .feature_engineering import FeatureEngineer
from ..utils.validators import validate_features

logger = logging.getLogger("profootball.models")


@dataclass
class PredictionResult:
    """Prediction result with confidence and metadata"""
    prediction: int  # 0 or 1
    probability: float
    confidence: str  # High/Medium/Low
    confidence_score: float
    feature_importance: Dict[str, float]
    risk_factors: List[str]
    expected_value: float
    
    @property
    def should_bet(self) -> bool:
        """Determine if this is a good betting opportunity"""
        return self.probability > 0.6 and self.expected_value > 0


class Over25Predictor:
    """Advanced ensemble model for Over 2.5 goals prediction"""
    
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_engineer = FeatureEngineer()
        self.feature_names = []
        self.model_path = DATA_DIR / "models" / f"{model_type}_model.pkl"
        self.is_trained = False
        
        # Model performance metrics
        self.performance_metrics = {
            "accuracy": 0.0,
            "auc_roc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
        
    def _create_ensemble_model(self) -> VotingClassifier:
        """Create advanced ensemble model"""
        models = []
        
        # Random Forest
        rf = RandomForestClassifier(
            **MODEL_CONFIG["ensemble"]["random_forest"],
            class_weight="balanced"
        )
        models.append(("rf", rf))
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(
            **MODEL_CONFIG["ensemble"]["gradient_boosting"]
        )
        models.append(("gb", gb))
        
        # XGBoost if available
        if XGBOOST_AVAILABLE:
            xgb_model = xgb.XGBClassifier(
                **MODEL_CONFIG["ensemble"]["xgboost"],
                use_label_encoder=False,
                eval_metric='logloss'
            )
            models.append(("xgb", xgb_model))
            
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft',
            n_jobs=-1
        )
        
        # Calibrate probabilities for better estimates
        return CalibratedClassifierCV(ensemble, method='sigmoid', cv=3)
        
    def train(self, X: pd.DataFrame, y: pd.Series, validate: bool = True) -> Dict[str, float]:
        """Train the model with advanced techniques"""
        logger.info(f"Training {self.model_type} model with {len(X)} samples")
        
        # Feature engineering
        X_engineered = self.feature_engineer.transform(X)
        self.feature_names = X_engineered.columns.tolist()
        
        # Validate features
        if not validate_features(X_engineered, self.feature_names):
            raise ValueError("Feature validation failed")
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X_engineered)
        
        # Create model
        if self.model_type == "ensemble":
            self.model = self._create_ensemble_model()
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(**MODEL_CONFIG["ensemble"]["random_forest"])
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(**MODEL_CONFIG["ensemble"]["gradient_boosting"])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        # Cross-validation
        if validate:
            cv = StratifiedKFold(n_splits=MODEL_CONFIG["training"]["cv_folds"], shuffle=True, random_state=42)
            
            scores = cross_val_score(
                self.model, X_scaled, y,
                cv=cv,
                scoring=MODEL_CONFIG["training"]["scoring"],
                n_jobs=-1
            )
            
            logger.info(f"Cross-validation {MODEL_CONFIG['training']['scoring']}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
            
        # Train final model
        self.model.fit(X_scaled, y)
        
        # Calculate performance metrics
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        self.performance_metrics = {
            "accuracy": (y_pred == y).mean(),
            "auc_roc": roc_auc_score(y, y_proba),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1_score": f1_score(y, y_pred)
        }
        
        self.is_trained = True
        
        # Save model
        self._save_model()
        
        logger.info(f"Model trained successfully. AUC-ROC: {self.performance_metrics['auc_roc']:.3f}")
        
        return self.performance_metrics
        
    def predict(self, features: Union[Dict, pd.DataFrame]) -> PredictionResult:
        """Make prediction with confidence assessment"""
        if not self.is_trained:
            self._load_model()
            
        # Convert to DataFrame if needed
        if isinstance(features, dict):
            features = pd.DataFrame([features])
            
        # Feature engineering
        X_engineered = self.feature_engineer.transform(features)
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(X_engineered.columns)
        if missing_features:
            # Add missing features with default values
            for feat in missing_features:
                X_engineered[feat] = 0
                
        # Reorder columns to match training
        X_engineered = X_engineered[self.feature_names]
        
        # Scale
        X_scaled = self.scaler.transform(X_engineered)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0, 1]
        
        # Get feature importance
        feature_importance = self._get_feature_importance(X_engineered)
        
        # Assess confidence
        confidence, confidence_score = self._assess_confidence(probability, X_engineered)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(X_engineered, feature_importance)
        
        # Calculate expected value (assuming odds of 1.85 for Over 2.5)
        expected_value = (probability * 1.85) - 1
        
        return PredictionResult(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence,
            confidence_score=float(confidence_score),
            feature_importance=feature_importance,
            risk_factors=risk_factors,
            expected_value=float(expected_value)
        )
        
    def batch_predict(self, features_list: List[Dict]) -> List[PredictionResult]:
        """Batch prediction for multiple matches"""
        df = pd.DataFrame(features_list)
        results = []
        
        # Process in batches for efficiency
        batch_size = 100
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            # Feature engineering for batch
            X_engineered = self.feature_engineer.transform(batch)
            
            # Ensure all features
            for feat in self.feature_names:
                if feat not in X_engineered.columns:
                    X_engineered[feat] = 0
                    
            X_engineered = X_engineered[self.feature_names]
            X_scaled = self.scaler.transform(X_engineered)
            
            # Predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            # Create results
            for j in range(len(batch)):
                idx = i + j
                
                result = PredictionResult(
                    prediction=int(predictions[j]),
                    probability=float(probabilities[j]),
                    confidence="Medium",  # Simplified for batch
                    confidence_score=0.7,
                    feature_importance={},
                    risk_factors=[],
                    expected_value=(probabilities[j] * 1.85) - 1
                )
                results.append(result)
                
        return results
        
    def _get_feature_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """Extract feature importance from model"""
        importance_dict = {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'estimator_') and hasattr(self.model.estimator_, 'feature_importances_'):
            # For calibrated models
            importances = self.model.estimator_.feature_importances_
        elif self.model_type == "ensemble" and hasattr(self.model, 'estimator_'):
            # Get average importance from ensemble
            importances = []
            for name, estimator in self.model.estimator_.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
                    
            if importances:
                importances = np.mean(importances, axis=0)
            else:
                return {}
        else:
            return {}
            
        # Create importance dictionary
        for feat, imp in zip(self.feature_names, importances):
            importance_dict[feat] = float(imp)
            
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
        
    def _assess_confidence(self, probability: float, features: pd.DataFrame) -> Tuple[str, float]:
        """Assess prediction confidence based on probability and features"""
        confidence_score = probability
        
        # Adjust based on probability distance from 0.5
        distance_from_half = abs(probability - 0.5)
        confidence_score *= (1 + distance_from_half)
        
        # Adjust based on feature quality
        if 'total_matches' in features.columns:
            matches = features['total_matches'].iloc[0]
            if matches < 10:
                confidence_score *= 0.8  # Lower confidence for small sample
            elif matches > 30:
                confidence_score *= 1.1  # Higher confidence for large sample
                
        # Normalize
        confidence_score = min(1.0, confidence_score)
        
        # Categorize
        if confidence_score >= 0.8:
            confidence = "High"
        elif confidence_score >= 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"
            
        return confidence, confidence_score
        
    def _identify_risk_factors(self, features: pd.DataFrame, importance: Dict[str, float]) -> List[str]:
        """Identify risk factors in the prediction"""
        risk_factors = []
        
        # Check for low sample size
        if 'total_matches' in features.columns and features['total_matches'].iloc[0] < 10:
            risk_factors.append("Low sample size (< 10 matches)")
            
        # Check for extreme values in important features
        top_features = list(importance.keys())[:5]
        
        for feat in top_features:
            if feat in features.columns:
                value = features[feat].iloc[0]
                
                # Define thresholds for common features
                if 'goals_avg' in feat and value < 0.5:
                    risk_factors.append(f"Very low {feat}: {value:.2f}")
                elif 'goals_avg' in feat and value > 3.5:
                    risk_factors.append(f"Unusually high {feat}: {value:.2f}")
                elif 'form' in feat and value < 3:
                    risk_factors.append(f"Poor recent form: {value}")
                    
        # Check for missing H2H data
        if 'h2h_over25_rate' in features.columns and features['h2h_over25_rate'].iloc[0] == 0:
            risk_factors.append("No H2H data available")
            
        return risk_factors[:3]  # Limit to top 3 risk factors
        
    def _save_model(self):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_engineer': self.feature_engineer,
            'performance_metrics': self.performance_metrics,
            'model_type': self.model_type,
            'trained_at': datetime.now().isoformat()
        }
        
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
    def _load_model(self):
        """Load model from disk"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        model_data = joblib.load(self.model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_engineer = model_data['feature_engineer']
        self.performance_metrics = model_data['performance_metrics']
        self.is_trained = True
        
        logger.info(f"Model loaded from {self.model_path}")
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metrics"""
        return {
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "features_count": len(self.feature_names),
            "performance_metrics": self.performance_metrics,
            "top_features": list(self._get_feature_importance(pd.DataFrame([{f: 0 for f in self.feature_names}])).keys())[:10]
        }
        
    def explain_prediction(self, features: Dict, prediction_result: PredictionResult) -> str:
        """Generate human-readable explanation for prediction"""
        explanation = []
        
        # Overall prediction
        if prediction_result.prediction == 1:
            explanation.append(f"✅ Model predicts Over 2.5 goals with {prediction_result.probability:.1%} probability")
        else:
            explanation.append(f"❌ Model predicts Under 2.5 goals with {(1-prediction_result.probability):.1%} probability")
            
        # Confidence
        explanation.append(f"Confidence: {prediction_result.confidence} ({prediction_result.confidence_score:.1%})")
        
        # Top factors
        top_factors = list(prediction_result.feature_importance.items())[:3]
        if top_factors:
            explanation.append("\nTop contributing factors:")
            for feat, imp in top_factors:
                feat_value = features.get(feat, 0)
                explanation.append(f"- {feat}: {feat_value:.2f} (importance: {imp:.3f})")
                
        # Risk factors
        if prediction_result.risk_factors:
            explanation.append("\n⚠️ Risk factors:")
            for risk in prediction_result.risk_factors:
                explanation.append(f"- {risk}")
                
        # Betting recommendation
        if prediction_result.should_bet:
            explanation.append(f"\n💰 Expected value: {prediction_result.expected_value:.3f} (Positive)")
        else:
            explanation.append(f"\n💰 Expected value: {prediction_result.expected_value:.3f} (Negative)")
            
        return "\n".join(explanation)