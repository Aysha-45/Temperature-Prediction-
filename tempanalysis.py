import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

class TemperaturePredictor:
    def __init__(self, lookback_period, forecast_horizon=10):
        self.lookback = lookback_period
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
        self.lr_model = LinearRegression()
        self.svr_model = SVR(kernel='rbf', C=100, gamma='auto')
        
    def create_features(self, df):
        X, y = [], []
        temp = df['Temperature'].values
        hum = df['Humidity'].values
        
        for i in range(len(df) - self.lookback - self.forecast_horizon):
            temp_features = temp[i : i + self.lookback]
            hum_features = hum[i : i + self.lookback]
            features = np.concatenate([temp_features, hum_features])
            label = temp[i + self.lookback + self.forecast_horizon - 1]
            
            X.append(features)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def train(self, X, y):
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                           test_size=0.2, 
                                                           shuffle=False)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        self.lr_model.fit(X_train_scaled, y_train)
        self.svr_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        y_pred_lr = self.lr_model.predict(X_test_scaled)
        y_pred_svr = self.svr_model.predict(X_test_scaled)
        
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        mse_svr = mean_squared_error(y_test, y_pred_svr)
        
        return mse_lr, mse_svr
    
    def predict(self, df):
        latest_temp = df['Temperature'].values[-self.lookback:]
        latest_hum = df['Humidity'].values[-self.lookback:]
        latest_features = np.concatenate([latest_temp, latest_hum]).reshape(1, -1)
        latest_features_scaled = self.scaler.transform(latest_features)
        
        next_temp_lr = self.lr_model.predict(latest_features_scaled)[0]
        next_temp_svr = self.svr_model.predict(latest_features_scaled)[0]
        
        return next_temp_lr, next_temp_svr

def main():
    # Read the data
    df = pd.read_csv('/Users/ayshaakmal/Downloads/SUMO_WORK/twentyfour.csv')
    
    # Initialize predictors with different lookback periods
    predictor_5min = TemperaturePredictor(lookback_period=5)
    predictor_50min = TemperaturePredictor(lookback_period=50)
    
    # Create features for both lookback periods
    X_5min, y_5min = predictor_5min.create_features(df)
    X_50min, y_50min = predictor_50min.create_features(df)
    
    # Train and evaluate models
    print("\n=== Results for 5-minute lookback ===")
    mse_lr_5min, mse_svr_5min = predictor_5min.train(X_5min, y_5min)
    print(f"Linear Regression MSE (5-min): {mse_lr_5min:.4f}")
    print(f"SVR MSE (5-min): {mse_svr_5min:.4f}")
    
    print("\n=== Results for 50-minute lookback ===")
    mse_lr_50min, mse_svr_50min = predictor_50min.train(X_50min, y_50min)
    print(f"Linear Regression MSE (50-min): {mse_lr_50min:.4f}")
    print(f"SVR MSE (50-min): {mse_svr_50min:.4f}")
    
    # Make predictions
    next_temp_lr_5min, next_temp_svr_5min = predictor_5min.predict(df)
    next_temp_lr_50min, next_temp_svr_50min = predictor_50min.predict(df)
    
    print("\n=== Temperature Predictions ===")
    print(f"Next 10-min Temperature (Linear Regression, 5-min lookback): {next_temp_lr_5min:.2f}")
    print(f"Next 10-min Temperature (SVR, 5-min lookback): {next_temp_svr_5min:.2f}")
    print(f"Next 10-min Temperature (Linear Regression, 50-min lookback): {next_temp_lr_50min:.2f}")
    print(f"Next 10-min Temperature (SVR, 50-min lookback): {next_temp_svr_50min:.2f}")
    

if __name__ == "__main__":
    main()
