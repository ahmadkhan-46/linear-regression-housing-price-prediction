import joblib
import numpy as np
import pandas as pd

def predict_price(input_data):
    
    lr = joblib.load('models/linear_regression.pkl')
    rf = joblib.load('models/random_forest.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    
    scaled_data = scaler.transform([input_data])
    
    
    lr_pred = lr.predict(scaled_data)
    rf_pred = rf.predict(scaled_data)
    
    return {
        'linear_regression': lr_pred[0],
        'random_forest': rf_pred[0],
        'average': (lr_pred[0] + rf_pred[0]) / 2
    }

if __name__ == "__main__":
    example_input = [0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.0900, 1, 296, 15.3, 396.90, 4.98]
    
    predictions = predict_price(example_input)
    print("Predicted Prices:")
    for model, price in predictions.items():
        print(f"{model.replace('_', ' ').title()}: ${price:,.2f}")