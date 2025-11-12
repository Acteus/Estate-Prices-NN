#!/usr/bin/env python3
"""
Philippine Property Price Prediction Neural Network

This notebook implements a deep learning model to predict property prices
in the Philippines using various property features and geographic coordinates.

Author: Kilo Code
Date: November 2025
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib

# Geospatial Libraries
from geopy.distance import geodesic

print("Libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Random seeds set for reproducibility")

"""
PHASE 1: DATA LOADING AND EXPLORATION
=====================================
"""

def load_and_explore_data(file_path='data/PH_houses_v2.csv'):
    """
    Load the PH houses dataset and perform initial exploration
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print("Loading Philippine Property Dataset...")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\n" + "="*50)
    
    # Display basic information
    print("DATASET OVERVIEW:")
    print(df.info())
    print("\n" + "="*50)
    
    print("FIRST 5 ROWS:")
    print(df.head())
    print("\n" + "="*50)
    
    print("DATASET DESCRIPTION:")
    print(df.describe())
    print("\n" + "="*50)
    
    # Check for missing values
    print("MISSING VALUES ANALYSIS:")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_percentage
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    print("\n" + "="*50)
    
    # Check for 'na' strings (not missing values but string 'na')
    print("STRING 'NA' VALUES ANALYSIS:")
    string_na_counts = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            na_count = (df[col] == 'na').sum()
            if na_count > 0:
                string_na_counts[col] = na_count
    
    if string_na_counts:
        for col, count in string_na_counts.items():
            print(f"{col}: {count} string 'na' values ({count/len(df)*100:.1f}%)")
    else:
        print("No string 'na' values found.")
    
    return df

"""
PHASE 2: DATA PREPROCESSING & FEATURE ENGINEERING
=================================================
"""

def clean_and_preprocess_data(df):
    """
    Clean and preprocess the dataset for neural network training
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed dataset
    """
    print("Starting data preprocessing...")
    
    df_clean = df.copy()
    
    # 1. Clean price data
    print("Cleaning price data...")
    df_clean['Price_Clean'] = df_clean['Price (PHP)'].astype(str).str.replace(',', '')
    
    def safe_convert_price(x):
        try:
            if str(x).lower() in ['na', 'nan', '']:
                return np.nan
            return float(x)
        except:
            return np.nan
    
    df_clean['Price_Clean'] = df_clean['Price_Clean'].apply(safe_convert_price)
    
    # 2. Clean numerical features
    print("Cleaning numerical features...")
    numerical_cols = ['Bedrooms', 'Bath', 'Floor_area (sqm)', 'Land_area (sqm)']
    
    for col in numerical_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 3. Clean geographic data
    print("Cleaning geographic data...")
    df_clean['Latitude'] = pd.to_numeric(df_clean['Latitude'], errors='coerce')
    df_clean['Longitude'] = pd.to_numeric(df_clean['Longitude'], errors='coerce')
    
    # 4. Extract property features from description
    print("Extracting property features from descriptions...")
    df_clean['Property_Type'] = extract_property_type(df_clean['Description'])
    df_clean['Bedrooms_from_Desc'] = extract_bedrooms_from_description(df_clean['Description'])
    df_clean['Has_Balcony'] = df_clean['Description'].str.contains('balcony', case=False, na=False).astype(int)
    df_clean['Is_Pre_Selling'] = df_clean['Description'].str.contains('pre-selling|pre selling', case=False, na=False).astype(int)
    df_clean['Is_RFO'] = df_clean['Description'].str.contains('rfo', case=False, na=False).astype(int)
    
    # 5. Geographic feature engineering
    print("Engineering geographic features...")
    manila_lat, manila_lon = 14.5995, 120.9842
    cebu_lat, cebu_lon = 10.3159, 123.8853
    davao_lat, davao_lon = 7.1907, 125.4553
    
    def calculate_distances(row):
        try:
            lat, lon = row['Latitude'], row['Longitude']
            if pd.isna(lat) or pd.isna(lon):
                return np.nan, np.nan, np.nan
            
            dist_manila = geodesic((manila_lat, manila_lon), (lat, lon)).kilometers
            dist_cebu = geodesic((cebu_lat, cebu_lon), (lat, lon)).kilometers
            dist_davao = geodesic((davao_lat, davao_lon), (lat, lon)).kilometers
            
            return dist_manila, dist_cebu, dist_davao
        except:
            return np.nan, np.nan, np.nan
    
    distances = df_clean.apply(calculate_distances, axis=1, result_type='expand')
    df_clean[['Dist_Manila', 'Dist_Cebu', 'Dist_Davao']] = distances
    
    # 6. Location feature engineering
    print("Engineering location features...")
    df_clean['City'] = extract_city_from_location(df_clean['Location'])
    df_clean['Province'] = extract_province_from_location(df_clean['Location'])
    df_clean['Region'] = map_to_region(df_clean['Province'])
    
    # 7. Create derived features
    print("Creating derived features...")
    df_clean['Floor_Area_per_Bedroom'] = df_clean['Floor_area (sqm)'] / (df_clean['Bedrooms'] + 0.1)  # Add small constant to avoid division by zero
    df_clean['Total_Rooms'] = df_clean['Bedrooms'] + df_clean['Bath']
    df_clean['Has_Land_Area'] = (~df_clean['Land_area (sqm)'].isna()).astype(int)
    
    # Log transform for skewed features
    for col in ['Floor_area (sqm)', 'Land_area (sqm)', 'Dist_Manila', 'Dist_Cebu', 'Dist_Davao']:
        if col in df_clean.columns:
            df_clean[f'{col}_log'] = np.log1p(df_clean[col])
    
    # 8. Handle missing values
    print("Handling missing values...")
    
    # For training data, we'll remove rows with missing target (price)
    df_clean = df_clean.dropna(subset=['Price_Clean']).reset_index(drop=True)
    
    # Fill missing values for numerical features with median
    numerical_features = ['Bedrooms', 'Bath', 'Floor_area (sqm)', 'Land_area (sqm)', 
                         'Dist_Manila', 'Dist_Cebu', 'Dist_Davao', 'Floor_Area_per_Bedroom', 'Total_Rooms']
    
    for col in numerical_features:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
    
    print(f"Dataset shape after preprocessing: {df_clean.shape}")
    print(f"Properties with valid prices: {len(df_clean)}")
    
    return df_clean

def extract_property_type(description):
    """Extract property type from description"""
    property_types = []
    
    for desc in description:
        desc_lower = str(desc).lower()
        
        if 'studio' in desc_lower:
            property_types.append('Studio')
        elif 'office' in desc_lower:
            property_types.append('Office')
        elif 'townhouse' in desc_lower:
            property_types.append('Townhouse')
        elif 'commercial' in desc_lower:
            property_types.append('Commercial')
        elif 'lot' in desc_lower or 'land' in desc_lower:
            property_types.append('Land')
        elif 'house' in desc_lower and 'condo' not in desc_lower:
            property_types.append('House')
        elif 'condo' in desc_lower:
            property_types.append('Condo')
        else:
            property_types.append('Other')
    
    return pd.Series(property_types)

def extract_bedrooms_from_description(description):
    """Extract bedroom count from description"""
    bedroom_counts = []
    
    for desc in description:
        desc_lower = str(desc).lower()
        
        # Look for bedroom patterns
        import re
        patterns = [
            r'(\d+)[-\s]*bedroom',
            r'(\d+)[-\s]*br',
            r'studio'  # Studio = 0 bedrooms
        ]
        
        bedroom_count = None
        for pattern in patterns:
            match = re.search(pattern, desc_lower)
            if match:
                if 'studio' in pattern:
                    bedroom_count = 0
                else:
                    bedroom_count = int(match.group(1))
                break
        
        if bedroom_count is None:
            bedroom_count = np.nan
        
        bedroom_counts.append(bedroom_count)
    
    return pd.Series(bedroom_counts)

def extract_city_from_location(location):
    """Extract city from location string"""
    cities = []
    
    for loc in location:
        loc_str = str(loc)
        # Common cities in Philippines
        city_keywords = {
            'cebu': 'Cebu City',
            'manila': 'Manila',
            'quezon': 'Quezon City',
            'makati': 'Makati',
            'taguig': 'Taguig',
            'pasig': 'Pasig',
            'mandaluyong': 'Mandaluyong',
            'caloocan': 'Caloocan',
            'paranaque': 'Paranaque',
            'las pinas': 'Las Pinas',
            'muntinlupa': 'Muntinlupa',
            'valenzuela': 'Valenzuela',
            'malabon': 'Malabon',
            'navotas': 'Navotas',
            'pasay': 'Pasay',
            'marikina': 'Marikina',
            'san juan': 'San Juan',
            ' Pateros': 'Pateros'
        }
        
        city = 'Unknown'
        loc_lower = loc_str.lower()
        
        for keyword, city_name in city_keywords.items():
            if keyword in loc_lower:
                city = city_name
                break
        
        cities.append(city)
    
    return pd.Series(cities)

def extract_province_from_location(location):
    """Extract province from location string"""
    provinces = []
    
    for loc in location:
        loc_str = str(loc)
        # Common provinces
        province_keywords = {
            'cebu': 'Cebu',
            'manila': 'Metro Manila',
            'cavite': 'Cavite',
            'laguna': 'Laguna',
            'batangas': 'Batangas',
            'rizal': 'Rizal',
            'bulacan': 'Bulacan',
            'pampanga': 'Pampanga',
            'bataan': 'Bataan',
            'tarlac': 'Tarlac',
            'nueva ecija': 'Nueva Ecija',
            'quezon': 'Quezon',
            'batangas': 'Batangas'
        }
        
        province = 'Unknown'
        loc_lower = loc_str.lower()
        
        for keyword, province_name in province_keywords.items():
            if keyword in loc_lower:
                province = province_name
                break
        
        provinces.append(province)
    
    return pd.Series(provinces)

def map_to_region(province):
    """Map province to region"""
    regions = []
    
    for prov in province:
        prov_str = str(prov).lower()
        
        # Luzon regions
        if prov_str in ['metro manila']:
            regions.append('NCR')
        elif prov_str in ['cavite', 'laguna', 'batangas', 'rizal', 'quezon']:
            regions.append('CALABARZON')
        elif prov_str in ['bulacan', 'pampanga', 'bataan', 'tarlac', 'nueva ecija']:
            regions.append('Central Luzon')
        
        # Visayas regions
        elif prov_str in ['cebu']:
            regions.append('Central Visayas')
        
        # Other
        else:
            regions.append('Other')
    
    return pd.Series(regions)

def prepare_features_for_model(df):
    """
    Prepare final feature set for neural network model
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        
    Returns:
        tuple: (X, y, feature_names, categorical_features, numerical_features)
    """
    print("Preparing features for neural network model...")
    
    # Define feature groups
    numerical_features = [
        'Bedrooms', 'Bath', 'Floor_area (sqm)', 'Land_area (sqm)',
        'Latitude', 'Longitude',
        'Dist_Manila', 'Dist_Cebu', 'Dist_Davao',
        'Floor_Area_per_Bedroom', 'Total_Rooms', 'Has_Land_Area',
        'Has_Balcony', 'Is_Pre_Selling', 'Is_RFO'
    ]
    
    categorical_features = [
        'Property_Type', 'City', 'Province', 'Region'
    ]
    
    # Select only available features
    available_numerical = [f for f in numerical_features if f in df.columns]
    available_categorical = [f for f in categorical_features if f in df.columns]
    
    print(f"Numerical features: {available_numerical}")
    print(f"Categorical features: {available_categorical}")
    
    # Prepare numerical features
    X_numerical = df[available_numerical].copy()
    
    # Prepare categorical features with label encoding
    X_categorical = pd.DataFrame(index=df.index)
    
    for col in available_categorical:
        le = LabelEncoder()
        X_categorical[col] = le.fit_transform(df[col].astype(str))
    
    # Target variable - ensure we only use rows with valid prices
    y = df['Price_Clean'].copy()
    
    # Ensure X and y have the same length (remove any rows where y is NaN)
    valid_mask = ~y.isna()
    X_numerical = X_numerical[valid_mask].reset_index(drop=True)
    X_categorical = X_categorical[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    
    # Check for NaN or Inf values before scaling
    print("Checking for NaN/Inf values in numerical features...")
    nan_count = X_numerical.isna().sum().sum()
    inf_count = np.isinf(X_numerical.select_dtypes(include=[np.number])).sum().sum()
    print(f"NaN values in numerical features: {nan_count}")
    print(f"Inf values in numerical features: {inf_count}")
    
    # Replace any remaining NaN or Inf with median values
    if nan_count > 0 or inf_count > 0:
        print("Replacing NaN/Inf values with median...")
        for col in X_numerical.columns:
            median_val = X_numerical[col].median()
            X_numerical[col] = X_numerical[col].replace([np.inf, -np.inf], median_val)
            X_numerical[col] = X_numerical[col].fillna(median_val)
    
    # Scale numerical features (important for neural networks)
    print("Scaling numerical features...")
    scaler_X = StandardScaler()
    X_numerical_scaled = pd.DataFrame(
        scaler_X.fit_transform(X_numerical),
        columns=X_numerical.columns,
        index=X_numerical.index
    )
    
    # Check for NaN/Inf after scaling
    nan_after = X_numerical_scaled.isna().sum().sum()
    inf_after = np.isinf(X_numerical_scaled.select_dtypes(include=[np.number])).sum().sum()
    if nan_after > 0 or inf_after > 0:
        print(f"WARNING: NaN/Inf values after scaling: NaN={nan_after}, Inf={inf_after}")
        # Replace with 0 (mean of scaled data)
        X_numerical_scaled = X_numerical_scaled.fillna(0)
        X_numerical_scaled = X_numerical_scaled.replace([np.inf, -np.inf], 0)
    
    # Scale target variable using log transformation (prices are typically log-normally distributed)
    print("Scaling target variable (log transformation)...")
    # Use log transformation for prices to handle large values
    y_log = np.log1p(y)  # log1p = log(1 + x) to handle zeros
    
    # Check for NaN/Inf in target
    if y_log.isna().sum() > 0 or np.isinf(y_log).sum() > 0:
        print("WARNING: NaN/Inf values in log-transformed target")
        y_log = y_log.replace([np.inf, -np.inf], y_log.median())
        y_log = y_log.fillna(y_log.median())
    
    scaler_y = StandardScaler()
    y_scaled = pd.Series(
        scaler_y.fit_transform(y_log.values.reshape(-1, 1)).flatten(),
        index=y.index
    )
    
    # Final check
    if y_scaled.isna().sum() > 0 or np.isinf(y_scaled).sum() > 0:
        print("WARNING: NaN/Inf values in scaled target")
        y_scaled = y_scaled.replace([np.inf, -np.inf], 0)
        y_scaled = y_scaled.fillna(0)
    
    # Combine all features for compatibility (if needed)
    X = pd.concat([X_numerical_scaled, X_categorical], axis=1)
    
    # Feature names for reference
    feature_names = list(X.columns)
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Numerical features shape: {X_numerical_scaled.shape}")
    print(f"Categorical features shape: {X_categorical.shape}")
    print(f"Target variable shape: {y_scaled.shape}")
    print(f"Price range (original): ₱{y.min():,.0f} - ₱{y.max():,.0f}")
    print(f"Price range (scaled): {y_scaled.min():.2f} - {y_scaled.max():.2f}")
    print(f"Features: {feature_names}")
    
    return (X, y_scaled, feature_names, available_categorical, available_numerical, 
            X_numerical_scaled, X_categorical, scaler_X, scaler_y, y)

"""
PHASE 3: NEURAL NETWORK ARCHITECTURE
=====================================
"""

def create_neural_network(input_dim, categorical_features, numerical_features, X_categorical=None, learning_rate=0.001):
    """
    Create neural network architecture for price prediction
    
    Args:
        input_dim (int): Number of input features
        categorical_features (list): List of categorical feature names
        numerical_features (list): List of numerical feature names
        X_categorical (pd.DataFrame): Categorical features dataframe to determine vocabulary sizes
        learning_rate (float): Learning rate for optimizer
        
    Returns:
        tf.keras.Model: Compiled neural network model
    """
    print("Creating neural network architecture...")
    
    # Input layers
    numerical_input = keras.Input(shape=(len(numerical_features),), name='numerical_input')
    categorical_inputs = {}
    
    # Create embedding layers for categorical features
    categorical_embeddings = []
    
    for feature in categorical_features:
        # Determine vocabulary size from actual data if available
        if X_categorical is not None and feature in X_categorical.columns:
            vocab_size = X_categorical[feature].max() + 1  # +1 because labels are 0-indexed
        else:
            vocab_size = 50  # Default estimate
        embedding_dim = min(max(vocab_size // 2, 10), 50)  # Between 10 and 50
        
        input_layer = keras.Input(shape=(1,), name=f'{feature}_input')
        embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
                                         name=f'{feature}_embedding')(input_layer)
        flatten_layer = layers.Flatten(name=f'{feature}_flatten')(embedding_layer)
        
        categorical_inputs[feature] = input_layer
        categorical_embeddings.append(flatten_layer)
    
    # Numerical input processing
    numerical_dense = layers.Dense(64, activation='relu', name='numerical_dense1')(numerical_input)
    numerical_dense = layers.Dropout(0.3)(numerical_dense)
    numerical_dense = layers.Dense(32, activation='relu', name='numerical_dense2')(numerical_dense)
    numerical_dense = layers.Dropout(0.2)(numerical_dense)
    
    # Combine all inputs
    if categorical_embeddings:
        combined = layers.Concatenate(name='concatenate')([numerical_dense] + categorical_embeddings)
    else:
        combined = numerical_dense
    
    # Hidden layers
    hidden1 = layers.Dense(128, activation='relu', name='hidden1')(combined)
    hidden1 = layers.Dropout(0.4)(hidden1)
    
    hidden2 = layers.Dense(64, activation='relu', name='hidden2')(hidden1)
    hidden2 = layers.Dropout(0.3)(hidden2)
    
    hidden3 = layers.Dense(32, activation='relu', name='hidden3')(hidden2)
    hidden3 = layers.Dropout(0.2)(hidden3)
    
    # Output layer
    output = layers.Dense(1, activation='linear', name='output')(hidden3)
    
    # Create model
    inputs = [numerical_input] + list(categorical_inputs.values())
    model = keras.Model(inputs=inputs, outputs=output, name='property_price_predictor')
    
    # Compile model with gradient clipping to prevent NaN during training
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)  # Clip gradients to prevent explosion
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    print("Neural network created successfully!")
    print(model.summary())
    
    return model

"""
PHASE 4: TRAINING AND VALIDATION
=================================
"""

def train_neural_network(model, X_train_num, X_train_cat, y_train, X_val_num, X_val_cat, y_val,
                        categorical_features, batch_size=32, epochs=100, patience=15):
    """
    Train the neural network model
    
    Args:
        model: Compiled neural network model
        X_train_num, X_train_cat: Training data (numerical and categorical)
        y_train: Training targets
        X_val_num, X_val_cat: Validation data (numerical and categorical)
        y_val: Validation targets
        categorical_features: List of categorical feature names
        batch_size (int): Batch size for training
        epochs (int): Maximum number of epochs
        patience (int): Patience for early stopping
        
    Returns:
        tuple: (trained_model, training_history)
    """
    print("Starting neural network training...")
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience//2,
        min_lr=0.00001,
        verbose=1
    )
    
    callbacks = [early_stopping, reduce_lr]
    
    # Prepare inputs: numerical features + separate categorical features
    X_train_num_vals = X_train_num.values.astype(np.float32)
    X_val_num_vals = X_val_num.values.astype(np.float32)
    
    # Check for NaN/Inf in training data
    if np.isnan(X_train_num_vals).any() or np.isinf(X_train_num_vals).any():
        print("WARNING: NaN/Inf found in training numerical features, replacing with 0")
        X_train_num_vals = np.nan_to_num(X_train_num_vals, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_num_vals = np.nan_to_num(X_val_num_vals, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train_inputs = [X_train_num_vals]
    X_val_inputs = [X_val_num_vals]
    
    # Add each categorical feature as a separate input
    for feature in categorical_features:
        if feature in X_train_cat.columns:
            cat_train = X_train_cat[feature].values.reshape(-1, 1).astype(np.int32)
            cat_val = X_val_cat[feature].values.reshape(-1, 1).astype(np.int32)
            X_train_inputs.append(cat_train)
            X_val_inputs.append(cat_val)
    
    # Check target for NaN/Inf
    y_train_vals = y_train.values.astype(np.float32)
    y_val_vals = y_val.values.astype(np.float32)
    
    if np.isnan(y_train_vals).any() or np.isinf(y_train_vals).any():
        print("WARNING: NaN/Inf found in training target, replacing with 0")
        y_train_vals = np.nan_to_num(y_train_vals, nan=0.0, posinf=0.0, neginf=0.0)
        y_val_vals = np.nan_to_num(y_val_vals, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Train the model
    history = model.fit(
        X_train_inputs, y_train_vals,
        validation_data=(X_val_inputs, y_val_vals),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("Training completed!")
    return model, history

if __name__ == "__main__":
    print("PHILIPPINE PROPERTY PRICE PREDICTION NEURAL NETWORK")
    print("=" * 60)
    
    # Phase 1: Data Loading and Exploration
    print("Phase 1: Data Loading and Exploration")
    df = load_and_explore_data()
    
    # Phase 2: Data Preprocessing
    print("\nPhase 2: Data Preprocessing and Feature Engineering")
    df_clean = clean_and_preprocess_data(df)
    
    # Phase 3: Feature Preparation
    print("\nPhase 3: Feature Preparation")
    X, y, feature_names, categorical_features, numerical_features = prepare_features_for_model(df_clean)
    
    # Phase 4: Model Creation and Training
    print("\nPhase 4: Neural Network Creation and Training")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42  # 0.25 x 0.8 = 0.2 overall
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create and train model
    model = create_neural_network(
        input_dim=X.shape[1],
        categorical_features=categorical_features,
        numerical_features=numerical_features
    )
    
    model, history = train_neural_network(model, X_train, y_train, X_val, y_val)
    
    print("Neural network training completed successfully!")
    print("Model is ready for evaluation and prediction.")