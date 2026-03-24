import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_model_file_exists():
    """Test that model file exists"""
    model_path = 'model/fraud_detection_complete.pkl'
    assert os.path.exists(model_path), f"Model file not found at {model_path}"

def test_model_loads():
    """Test that model can be loaded"""
    import pickle
    try:
        with open('model/fraud_detection_complete.pkl', 'rb') as f:
            package = pickle.load(f)
        assert package is not None, "Model package is None"
        assert 'model' in package, "Model not in package"
        assert 'scaler' in package, "Scaler not in package"
        assert 'features' in package, "Features not in package"
    except Exception as e:
        pytest.fail(f"Model failed to load: {str(e)}")

def test_required_files_exist():
    """Test that all required files exist"""
    required_files = [
        'app.py',
        'database.py',
        'model_version.py',
        'monitoring.py',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    assert len(missing_files) == 0, f"Missing files: {missing_files}"

def test_templates_exist():
    """Test that template files exist"""
    templates = [
        'templates/index.html',
        'templates/history.html',
        'templates/metrics.html',
        'templates/monitoring.html'
    ]
    
    missing_templates = [t for t in templates if not os.path.exists(t)]
    assert len(missing_templates) == 0, f"Missing templates: {missing_templates}"

def test_model_features():
    """Test that model has correct features"""
    import pickle
    with open('model/fraud_detection_complete.pkl', 'rb') as f:
        package = pickle.load(f)
    
    expected_features = [
        'amount',
        'hour',
        'is_night',
        'is_high_amount',
        'transaction_type_enc',
        'merchant_category_enc',
        'country_enc'
    ]
    
    actual_features = package['features']
    assert actual_features == expected_features, f"Features mismatch. Expected: {expected_features}, Got: {actual_features}"

def test_encoders_exist():
    """Test that label encoders exist in model package"""
    import pickle
    with open('model/fraud_detection_complete.pkl', 'rb') as f:
        package = pickle.load(f)
    
    assert 'le_transaction' in package, "Transaction encoder missing"
    assert 'le_merchant' in package, "Merchant encoder missing"
    assert 'le_country' in package, "Country encoder missing"

def test_model_metrics_file():
    """Test that model metrics file exists and is valid JSON"""
    import json
    assert os.path.exists('model_metrics.json'), "model_metrics.json not found"
    
    try:
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)
        assert 'models' in metrics, "models key missing in metrics"
        assert len(metrics['models']) > 0, "No models in metrics"
    except json.JSONDecodeError:
        pytest.fail("model_metrics.json is not valid JSON")

def test_dockerfile_exists():
    """Test that Dockerfile exists and contains Python"""
    assert os.path.exists('Dockerfile'), "Dockerfile not found"
    
    with open('Dockerfile', 'r') as f:
        content = f.read()
    
    # Check for Python base image (case-insensitive)
    content_lower = content.lower()
    assert 'from python' in content_lower, "Dockerfile doesn't use Python base image"
    assert 'copy requirements.txt' in content_lower, "Dockerfile doesn't copy requirements.txt"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])