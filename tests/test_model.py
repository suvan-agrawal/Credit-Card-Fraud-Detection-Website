import pytest
import pickle
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def model_package():
    """Load model package for testing"""
    with open('model/fraud_detection_complete.pkl', 'rb') as f:
        return pickle.load(f)

def test_model_prediction(model_package):
    """Test that model can make predictions"""
    model = model_package['model']
    scaler = model_package['scaler']
    features = model_package['features']
    
    # Create test input
    test_data = pd.DataFrame([[
        1500.0,  # amount
        14,      # hour
        0,       # is_night
        1,       # is_high_amount
        0,       # transaction_type_enc
        1,       # merchant_category_enc
        0        # country_enc
    ]], columns=features)
    
    # Scale
    test_data[['amount', 'hour']] = scaler.transform(test_data[['amount', 'hour']])
    
    # Predict
    prediction = model.predict(test_data)
    probability = model.predict_proba(test_data)
    
    assert prediction is not None, "Prediction is None"
    assert len(probability) == 1, f"Expected 1 prediction, got {len(probability)}"
    assert len(probability[0]) == 2, f"Expected 2 probabilities, got {len(probability[0])}"
    assert 0 <= probability[0][1] <= 1, f"Probability out of range: {probability[0][1]}"

def test_model_type(model_package):
    """Test that model is Random Forest"""
    from sklearn.ensemble import RandomForestClassifier
    model = model_package['model']
    assert isinstance(model, RandomForestClassifier), f"Model is {type(model)}, not Random Forest"

def test_scaler_type(model_package):
    """Test that scaler is StandardScaler"""
    from sklearn.preprocessing import StandardScaler
    scaler = model_package['scaler']
    assert isinstance(scaler, StandardScaler), f"Scaler is {type(scaler)}, not StandardScaler"

def test_label_encoders(model_package):
    """Test label encoders have correct classes"""
    le_transaction = model_package['le_transaction']
    le_merchant = model_package['le_merchant']
    le_country = model_package['le_country']
    
    # Check transaction types
    assert len(le_transaction.classes_) == 4, f"Expected 4 transaction types, got {len(le_transaction.classes_)}"
    assert 'ATM' in le_transaction.classes_, "ATM not in transaction types"
    assert 'Online' in le_transaction.classes_, "Online not in transaction types"
    
    # Check merchant categories
    assert len(le_merchant.classes_) == 5, f"Expected 5 merchant categories, got {len(le_merchant.classes_)}"
    assert 'Electronics' in le_merchant.classes_, "Electronics not in merchant categories"
    
    # Check countries
    assert len(le_country.classes_) == 6, f"Expected 6 countries, got {len(le_country.classes_)}"

def test_high_fraud_prediction(model_package):
    """Test prediction for high-risk transaction"""
    model = model_package['model']
    scaler = model_package['scaler']
    features = model_package['features']
    
    # High-risk transaction: Large amount, night time, risky country
    test_data = pd.DataFrame([[
        5000.0,  # amount - very high
        3,       # hour - night time
        1,       # is_night
        1,       # is_high_amount
        0,       # transaction_type_enc (ATM)
        3,       # merchant_category_enc (Travel)
        4        # country_enc (NG - risky)
    ]], columns=features)
    
    # Scale
    test_data[['amount', 'hour']] = scaler.transform(test_data[['amount', 'hour']])
    
    # Predict
    probability = model.predict_proba(test_data)[0][1]
    
    # Should have high fraud probability
    assert probability > 0.5, f"High-risk transaction has low fraud probability: {probability}"

def test_low_fraud_prediction(model_package):
    """Test prediction for low-risk transaction"""
    model = model_package['model']
    scaler = model_package['scaler']
    features = model_package['features']
    
    # Low-risk transaction: Small amount, daytime, safe country
    test_data = pd.DataFrame([[
        50.0,    # amount - small
        14,      # hour - afternoon
        0,       # is_night - no
        0,       # is_high_amount - no
        2,       # transaction_type_enc (POS)
        2,       # merchant_category_enc (Grocery)
        5        # country_enc (US - safe)
    ]], columns=features)
    
    # Scale
    test_data[['amount', 'hour']] = scaler.transform(test_data[['amount', 'hour']])
    
    # Predict
    probability = model.predict_proba(test_data)[0][1]
    
    # Should have low fraud probability
    assert probability < 0.5, f"Low-risk transaction has high fraud probability: {probability}"

def test_model_performance_metrics(model_package):
    """Test that model has reasonable performance"""
    # This assumes your model was trained with good metrics
    # You can add assertions based on your model's actual performance
    model = model_package['model']
    
    # Check that model has been fitted
    assert hasattr(model, 'n_features_in_'), "Model hasn't been fitted"
    assert model.n_features_in_ == 7, f"Model expects {model.n_features_in_} features, should be 7"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])