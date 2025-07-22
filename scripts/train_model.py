import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import load_and_preprocess_data

def train_and_evaluate_models():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    lr_mse = mean_squared_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)
    
    print("Linear Regression:")
    print(f"MSE: {lr_mse:.2f}, R2 Score: {lr_r2:.2f}")
    
   
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    print("\nRandom Forest Regression:")
    print(f"MSE: {rf_mse:.2f}, R2 Score: {rf_r2:.2f}")
    
   
    joblib.dump(lr, 'models/linear_regression.pkl')
    joblib.dump(rf, 'models/random_forest.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return lr, rf

if __name__ == "__main__":
    train_and_evaluate_models()