import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from tensorflow.keras import layers
import google.generativeai as genai
import streamlit as st
import joblib
from datetime import datetime, timedelta

# Configure Gemini AI
genai.configure(api_key=st.secrets["AIzaSyBMopZ2k-O9PZQLVEQQV36r8wRAlJBsT-o"])
model = genai.GenerativeModel('gemini-pro')

class AccountingAISystem:
    def __init__(self):
        self.rf_model = None
        self.dl_model = None
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self, start_date='2020-01-01', periods=48):
        """
        Generate synthetic financial data
        """
        np.random.seed(42)
        dates = pd.date_range(start=start_date, periods=periods, freq='M')
        
        # Base values
        base_revenue = 1000000
        base_expenses = 800000
        base_assets = 5000000
        base_liabilities = 3000000
        
        # Add trend and seasonality
        trend = np.linspace(0, 0.5, periods)  # Upward trend
        seasonality = np.sin(np.linspace(0, 4*np.pi, periods))  # Seasonal pattern
        
        data = pd.DataFrame({
            'date': dates,
            'revenue': base_revenue * (1 + trend + 0.1 * seasonality) * np.random.normal(1, 0.05, periods),
            'expenses': base_expenses * (1 + 0.8 * trend + 0.08 * seasonality) * np.random.normal(1, 0.04, periods),
            'assets': base_assets * (1 + trend + 0.05 * seasonality) * np.random.normal(1, 0.03, periods),
            'liabilities': base_liabilities * (1 + 0.7 * trend + 0.05 * seasonality) * np.random.normal(1, 0.03, periods)
        })
        
        # Calculate derived metrics
        data['profit'] = data['revenue'] - data['expenses']
        data['net_worth'] = data['assets'] - data['liabilities']
        data['profit_margin'] = (data['profit'] / data['revenue']) * 100
        
        return data
        
    def prepare_data(self, financial_data):
        """
        Prepare financial data for ML/DL models
        """
        # Convert dates to numerical features
        financial_data['year'] = pd.to_datetime(financial_data['date']).dt.year
        financial_data['month'] = pd.to_datetime(financial_data['date']).dt.month
        
        # Normalize numerical columns
        numerical_cols = ['revenue', 'expenses', 'assets', 'liabilities']
        financial_data[numerical_cols] = self.scaler.fit_transform(financial_data[numerical_cols])
        
        return financial_data
    
    def build_ml_model(self, X_train, y_train):
        """
        Build and train Random Forest model for financial predictions
        """
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.rf_model.fit(X_train, y_train)
        return self.rf_model.score(X_train, y_train)
        
    def build_dl_model(self, input_shape):
        """
        Build and compile deep learning model for complex financial patterns
        """
        self.dl_model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=input_shape),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        self.dl_model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
    
    def train_dl_model(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Train the deep learning model
        """
        return self.dl_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
    
    def get_gemini_response(self, query):
        """
        Get response from Gemini AI for accounting-related queries
        """
        try:
            response = model.generate_content(
                f"As an accounting expert, please answer this question: {query}"
            )
            return response.text
        except Exception as e:
            return f"Error getting Gemini response: {str(e)}"
    
    def save_models(self, path):
        """
        Save trained models and scaler
        """
        joblib.dump(self.rf_model, f"{path}/rf_model.joblib")
        self.dl_model.save(f"{path}/dl_model.h5")
        joblib.dump(self.scaler, f"{path}/scaler.joblib")
        
    def load_models(self, path):
        """
        Load trained models and scaler
        """
        self.rf_model = joblib.load(f"{path}/rf_model.joblib")
        self.dl_model = keras.models.load_model(f"{path}/dl_model.h5")
        self.scaler = joblib.load(f"{path}/scaler.joblib")

def create_web_interface():
    st.title("AI-Powered Accounting System")
    
    # Initialize system
    system = AccountingAISystem()
    
    # Generate synthetic data
    data = system.generate_synthetic_data()
    
    tab1, tab2, tab3 = st.tabs(["ML Predictions", "DL Analysis", "Gemini AI Assistant"])
    
    with tab1:
        st.header("Machine Learning Financial Predictions")
        st.subheader("Sample Financial Data")
        st.dataframe(data.head())
        
        if st.button("Train ML Model"):
            # Prepare data
            prepared_data = system.prepare_data(data.copy())
            X = prepared_data[['year', 'month', 'revenue', 'expenses']]
            y = prepared_data['profit']
            
            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            score = system.build_ml_model(X_train, y_train)
            st.success(f"ML Model trained successfully! R² Score: {score:.4f}")
    
    with tab2:
        st.header("Deep Learning Financial Analysis")
        if st.button("Train DL Model"):
            # Prepare data
            prepared_data = system.prepare_data(data.copy())
            X = prepared_data[['year', 'month', 'revenue', 'expenses']]
            y = prepared_data['profit']
            
            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            system.build_dl_model((X_train.shape[1],))
            history = system.train_dl_model(X_train, y_train)
            st.success("DL Model trained successfully!")
            
            # Plot training history
            st.line_chart(pd.DataFrame(history.history))
    
    with tab3:
        st.header("Gemini AI Accounting Assistant")
        query = st.text_input("Ask your accounting question:")
        if st.button("Get Answer"):
            response = system.get_gemini_response(query)
            st.write(response)

if __name__ == "__main__":
    create_web_interface()
