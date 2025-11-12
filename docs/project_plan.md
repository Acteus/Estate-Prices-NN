# Philippine Property Price Prediction Neural Network Project

## Project Overview
Create a deep learning model to predict property prices in the Philippines using the PH_houses_v2.csv dataset with features like location, property characteristics, and geographic coordinates.

## Dataset Analysis Summary
**Dataset**: PH_houses_v2.csv (1,500+ properties)
**Features**:
- **Target Variable**: Price (PHP) - numeric values ranging from ~850K to 87M PHP
- **Property Features**: Bedrooms, Bathrooms, Floor_area, Land_area  
- **Location Features**: Description, Location, Latitude, Longitude
- **Metadata**: Link (not used for prediction)

**Key Challenges**:
- Mixed data types (numerical, categorical, geographic)
- Missing values ("na" entries)
- Large price range and potential outliers
- Location text encoding and geographic feature engineering

## Project Workflow

### Phase 1: Data Understanding & Preparation
- Exploratory Data Analysis (EDA)
- Missing value analysis and treatment
- Price distribution analysis and outlier detection
- Feature correlation analysis

### Phase 2: Feature Engineering  
- Text processing for location and property descriptions
- Geographic feature engineering from coordinates
- Categorical encoding for property types and locations
- Numerical feature normalization/standardization
- New feature creation (price per sqm, etc.)

### Phase 3: Model Architecture
- Neural network design for regression task
- Input preprocessing pipeline
- Mixed input handling (dense, embedding layers)
- Regularization and dropout strategies

### Phase 4: Implementation & Training
- Framework selection (TensorFlow/PyTorch)
- Data splitting strategy (70/15/15)
- Training pipeline with validation
- Hyperparameter optimization
- Early stopping and model checkpointing

### Phase 5: Evaluation & Analysis
- Performance metrics (MAE, RMSE, R², MAPE)
- Prediction visualization
- Feature importance analysis
- Performance across property types and locations

### Phase 6: Documentation & Deployment
- Model documentation
- Training pipeline code
- Prediction pipeline for new data
- Performance report

## Expected Deliverables
1. **Jupyter Notebook**: Complete analysis and model training
2. **Trained Model**: Saved neural network model
3. **Prediction Pipeline**: Code for making predictions on new data
4. **Performance Report**: Detailed model evaluation and insights
5. **Documentation**: Technical documentation and usage guide

## Success Metrics
- **Primary**: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)
- **Secondary**: R² Score and Mean Absolute Percentage Error (MAPE)
- **Target**: Achieve <20% MAPE on test set across different property types

## Timeline Estimate
- **Data Analysis**: 1-2 days
- **Feature Engineering**: 1-2 days  
- **Model Development**: 2-3 days
- **Training & Optimization**: 1-2 days
- **Evaluation & Documentation**: 1 day

**Total Estimated Time**: 6-10 days