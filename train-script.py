#!/usr/bin/env python
"""
Script to train or retrain the prediction models
"""

import asyncio
import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.predictor import Over25Predictor
from src.models.feature_engineering import FeatureEngineer
from src.data.database import DatabaseManager
from src.utils.logger import setup_logging
from config import DATA_DIR, MODEL_CONFIG

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


async def train_models(model_type: str = "ensemble", n_samples: int = 5000):
    """Train prediction models"""
    
    logger.info(f"Starting model training - Type: {model_type}, Samples: {n_samples}")
    
    # Initialize database
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    try:
        # Create predictor
        predictor = Over25Predictor(model_type=model_type)
        
        # Get training data from database
        logger.info("Loading training data...")
        training_data = await db_manager.get_training_data()
        
        if training_data is None or len(training_data) < 100:
            logger.info("Insufficient historical data, generating synthetic data...")
            # Generate synthetic training data
            from src.models.predictor import Over25Predictor
            temp_predictor = Over25Predictor()
            training_data = temp_predictor.create_training_data(n_samples=n_samples)
        
        # Prepare features and target
        feature_cols = MODEL_CONFIG['features']
        X = training_data[feature_cols]
        y = training_data['over25']
        
        logger.info(f"Training with {len(X)} samples...")
        
        # Train model
        metrics = predictor.train(X, y, validate=True)
        
        # Log results
        logger.info("Training completed successfully!")
        logger.info(f"Model metrics:")
        for metric, value in metrics.items():
            logger.info(f"  - {metric}: {value:.3f}")
        
        # Save model info
        model_info = predictor.get_model_info()
        logger.info(f"Model saved to: {predictor.model_path}")
        
        # Test prediction
        logger.info("Testing prediction...")
        test_features = {col: 1.0 for col in feature_cols}
        test_result = predictor.predict(test_features)
        logger.info(f"Test prediction: {test_result.probability:.2%} (Over 2.5)")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return False
        
    finally:
        await db_manager.close()


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Train ProFootballAI models")
    parser.add_argument(
        "--model-type",
        type=str,
        default="ensemble",
        choices=["ensemble", "random_forest", "gradient_boosting"],
        help="Type of model to train"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of synthetic samples to generate if needed"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if model exists"
    )
    
    args = parser.parse_args()
    
    # Check if model already exists
    model_path = DATA_DIR / "models" / f"{args.model_type}_model.pkl"
    
    if model_path.exists() and not args.force:
        logger.info(f"Model already exists at {model_path}")
        response = input("Do you want to retrain? (y/N): ")
        if response.lower() != 'y':
            logger.info("Training cancelled")
            return
    
    # Run training
    success = asyncio.run(train_models(args.model_type, args.samples))
    
    if success:
        logger.info("✅ Model training completed successfully!")
    else:
        logger.error("❌ Model training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()