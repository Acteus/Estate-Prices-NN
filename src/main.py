"""
MAIN EXECUTION SCRIPT - Philippine Property Price Prediction Neural Network
============================================================================

This script runs the complete pipeline from data loading to model deployment.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (parent of src/)
project_root = os.path.dirname(script_dir)
# Add project root and src to path
sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)

# Import custom modules
from property_price_prediction import *
from evaluation_and_deployment import *

def main():
    """
    Main execution function for the Neural Network Property Price Prediction
    """
    # Change to project root directory for relative paths
    os.chdir(project_root)
    
    print("🚀 PHILIPPINE PROPERTY PRICE PREDICTION NEURAL NETWORK")
    print("=" * 70)
    print("Starting complete pipeline execution...")
    print("=" * 70)
    
    # Create output directories (relative to project root)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    try:
        # PHASE 1: DATA LOADING AND EXPLORATION
        print("\n📊 PHASE 1: DATA LOADING AND EXPLORATION")
        print("-" * 50)
        
        # Load and explore data
        df = load_and_explore_data('data/PH_houses_v2.csv')
        
        # PHASE 2: DATA PREPROCESSING AND FEATURE ENGINEERING
        print("\n🔧 PHASE 2: DATA PREPROCESSING AND FEATURE ENGINEERING")
        print("-" * 50)
        
        df_clean = clean_and_preprocess_data(df)
        
        # PHASE 3: FEATURE PREPARATION
        print("\n📋 PHASE 3: FEATURE PREPARATION")
        print("-" * 50)
        
        (X, y, feature_names, categorical_features, numerical_features, 
         X_numerical, X_categorical, scaler_X, scaler_y, y_original) = prepare_features_for_model(df_clean)
        
        # PHASE 4: DATA SPLITTING
        print("\n📊 PHASE 4: DATA SPLITTING")
        print("-" * 50)
        
        # Split data into train/validation/test sets (using indices to keep numerical and categorical aligned)
        indices = X_numerical.index
        train_indices, test_indices = train_test_split(
            indices, test_size=0.2, random_state=42
        )
        train_indices, val_indices = train_test_split(
            train_indices, test_size=0.25, random_state=42
        )
        
        # Split numerical and categorical features separately
        X_train_num = X_numerical.loc[train_indices].reset_index(drop=True)
        X_val_num = X_numerical.loc[val_indices].reset_index(drop=True)
        X_test_num = X_numerical.loc[test_indices].reset_index(drop=True)
        
        X_train_cat = X_categorical.loc[train_indices].reset_index(drop=True)
        X_val_cat = X_categorical.loc[val_indices].reset_index(drop=True)
        X_test_cat = X_categorical.loc[test_indices].reset_index(drop=True)
        
        y_train = y.loc[train_indices].reset_index(drop=True)
        y_val = y.loc[val_indices].reset_index(drop=True)
        y_test = y.loc[test_indices].reset_index(drop=True)
        
        # Also create combined X for evaluation functions that might need it
        X_train = pd.concat([X_train_num, X_train_cat], axis=1)
        X_val = pd.concat([X_val_num, X_val_cat], axis=1)
        X_test = pd.concat([X_test_num, X_test_cat], axis=1)
        
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Validation set size: {X_val.shape[0]} samples")
        print(f"Test set size: {X_test.shape[0]} samples")
        print(f"Total features: {X.shape[1]}")
        
        # PHASE 5: NEURAL NETWORK MODEL CREATION
        print("\n🧠 PHASE 5: NEURAL NETWORK MODEL CREATION")
        print("-" * 50)
        
        # Create neural network model
        model = create_neural_network(
            input_dim=X.shape[1],
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            X_categorical=X_categorical,
            learning_rate=0.001
        )
        
        # PHASE 6: MODEL TRAINING
        print("\n🎯 PHASE 6: MODEL TRAINING")
        print("-" * 50)
        
        trained_model, history = train_neural_network(
            model, X_train_num, X_train_cat, y_train, X_val_num, X_val_cat, y_val,
            categorical_features, batch_size=32, epochs=100, patience=15
        )
        
        # PHASE 7: MODEL EVALUATION
        print("\n📈 PHASE 7: MODEL EVALUATION")
        print("-" * 50)
        
        # Evaluate on test set (need original y_test for inverse transform)
        y_test_original = y_original.loc[test_indices].reset_index(drop=True)
        metrics, y_pred = evaluate_model_performance(
            trained_model, X_test_num, X_test_cat, y_test, y_test_original, 
            feature_names, categorical_features, scaler_y
        )
        
        # Create evaluation plots
        plot_path = create_evaluation_plots(y_test_original, y_pred, history, save_path='output/Figure_1.png')
        print(f"✅ Evaluation plots saved to: {plot_path}")
        
        # PHASE 8: FEATURE IMPORTANCE ANALYSIS
        print("\n🔍 PHASE 8: FEATURE IMPORTANCE ANALYSIS")
        print("-" * 50)
        
        importance_df = analyze_feature_importance(df_clean, X, y, feature_names)
        
        # PHASE 9: PERFORMANCE BY SEGMENTS
        print("\n📊 PHASE 9: PERFORMANCE BY PROPERTY SEGMENTS")
        print("-" * 50)
        
        segment_results = performance_by_segments(df_clean, y_test, y_pred, X_test)
        
        # PHASE 10: MODEL DEPLOYMENT SETUP
        print("\n💾 PHASE 10: MODEL DEPLOYMENT SETUP")
        print("-" * 50)
        
        # Create predictor instance
        predictor = PropertyPricePredictor(
            model=trained_model,
            scalers=None,  # Would be filled with actual scalers
            feature_names=feature_names
        )
        
        # Save model
        predictor.save_model('models/property_price_model')
        
        # Create prediction examples
        predictor, examples = create_prediction_examples()
        
        # PHASE 11: GENERATE FINAL REPORT
        print("\n📄 PHASE 11: GENERATING FINAL REPORT")
        print("-" * 50)
        
        model_info = {
            'architecture': 'Neural Network',
            'framework': 'TensorFlow/Keras',
            'layers': len(trained_model.layers),
            'parameters': trained_model.count_params()
        }
        
        report = generate_model_report(metrics, importance_df, segment_results, model_info)
        
        # Save detailed report
        with open('reports/final_model_report.txt', 'w') as f:
            f.write(report)
        
        print("\n" + "=" * 70)
        print("✅ NEURAL NETWORK PROPERTY PRICE PREDICTION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📁 Generated Files:")
        print("- models/property_price_model_nn.h5 (Trained Neural Network)")
        print("- models/property_price_model_preprocessing.pkl (Preprocessing Objects)")
        print("- reports/final_model_report.txt (Performance Report)")
        print("- plots/ (Evaluation Visualizations)")
        
        print(f"\n🎯 Final Performance Summary:")
        print(f"- Mean Absolute Error: ₱{metrics['MAE']:,.0f}")
        print(f"- R² Score: {metrics['R²']:.4f} ({metrics['R²']:.1%} accuracy)")
        print(f"- MAPE: {metrics['MAPE']:.2f}%")
        print(f"- Model Parameters: {trained_model.count_params():,}")
        
        if metrics['MAPE'] < 20:
            print("\n🌟 EXCELLENT! The model achieves good accuracy for property price prediction.")
        elif metrics['MAPE'] < 30:
            print("\n👍 GOOD! The model provides reasonable price estimates.")
        else:
            print("\n⚠️  The model needs improvement. Consider collecting more data or tuning hyperparameters.")
        
        print("\n🚀 Your Neural Network Property Price Prediction Model is ready for deployment!")
        
        return trained_model, metrics, importance_df, segment_results
        
    except Exception as e:
        print(f"\n❌ ERROR: An error occurred during execution: {str(e)}")
        print("\nTroubleshooting suggestions:")
        print("1. Check if data/PH_houses_v2.csv exists in the project root")
        print("2. Ensure all required libraries are installed (see requirements.txt)")
        print("3. Verify system has sufficient memory for neural network training")
        print("4. Check if TensorFlow and other ML libraries are properly installed")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())
        return None

def quick_demo():
    """
    Quick demo version with reduced training for testing
    """
    # Change to project root directory for relative paths
    os.chdir(project_root)
    
    print("🚀 QUICK DEMO - PHILIPPINE PROPERTY PRICE PREDICTION")
    print("=" * 60)
    
    try:
        # Load data
        df = load_and_explore_data('data/PH_houses_v2.csv')
        df_clean = clean_and_preprocess_data(df)
        (X, y, feature_names, categorical_features, numerical_features, 
         X_numerical, X_categorical, scaler_X, scaler_y, y_original) = prepare_features_for_model(df_clean)
        
        # Quick split (using indices to keep numerical and categorical aligned)
        indices = X_numerical.index
        train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=42)
        
        X_train_num = X_numerical.loc[train_indices].reset_index(drop=True)
        X_test_num = X_numerical.loc[test_indices].reset_index(drop=True)
        X_train_cat = X_categorical.loc[train_indices].reset_index(drop=True)
        X_test_cat = X_categorical.loc[test_indices].reset_index(drop=True)
        y_train = y.loc[train_indices].reset_index(drop=True)
        y_test = y.loc[test_indices].reset_index(drop=True)
        y_test_original = y_original.loc[test_indices].reset_index(drop=True)
        
        # Simple neural network
        model = create_neural_network(
            input_dim=X.shape[1],
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            X_categorical=X_categorical,
            learning_rate=0.01  # Higher learning rate for faster training
        )
        
        # Quick training (using test as validation for quick demo)
        trained_model, history = train_neural_network(
            model, X_train_num, X_train_cat, y_train, X_test_num, X_test_cat, y_test,
            categorical_features, batch_size=64, epochs=50, patience=10
        )
        
        # Quick evaluation
        metrics, y_pred = evaluate_model_performance(
            trained_model, X_test_num, X_test_cat, y_test, y_test_original, 
            feature_names, categorical_features, scaler_y
        )
        
        print("\n🎯 Quick Demo Results:")
        print(f"- R² Score: {metrics['R²']:.4f}")
        print(f"- MAPE: {metrics['MAPE']:.2f}%")
        print("\n✅ Quick demo completed successfully!")
        
        return trained_model
        
    except Exception as e:
        print(f"❌ Quick demo failed: {str(e)}")
        return None

if __name__ == "__main__":
    # Check if running in demo mode
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        print("Running in DEMO mode (reduced training)...")
        model = quick_demo()
    else:
        print("Running in FULL mode (complete pipeline)...")
        model = main()