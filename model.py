def predict(features):
    """
    A mock prediction function that simulates a model prediction.
    
    Args:
        features (list): A list of input features.

    Returns:
        float: A simulated prediction value.
    """
    # Example logic for prediction (replace with your actual model logic)
    if len(features) != 3:
        raise ValueError("Expected 3 features for prediction.")
    
    # Simple mock prediction logic (for demonstration purposes)
    prediction = sum(features) / len(features)  # Average of the features
    return prediction