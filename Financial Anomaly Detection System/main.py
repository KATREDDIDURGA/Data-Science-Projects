"""
Financial Anomaly Detection System
Real-time anomaly detection for financial transactions using advanced ML algorithms

Author: Durga Katreddi
Email: katreddisrisaidurga@gmail.com
LinkedIn: https://linkedin.com/in/sri-sai-durga-katreddi-
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam
import torch
import torch.nn as nn

# Time Series Analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Database and Cloud
import pymongo
from sqlalchemy import create_engine
import redis
from azure.storage.blob import BlobServiceClient
from azure.servicebus import ServiceBusClient

# API and Deployment
from flask import Flask, request, jsonify
import docker
import kubernetes

# Monitoring and Alerting
import smtplib
from email.mime.text import MIMEText
import slack_sdk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialAnomalyDetector:
    """
    Advanced Financial Anomaly Detection System
    
    Features:
    - Real-time transaction monitoring
    - Multiple anomaly detection algorithms
    - Pattern recognition and behavioral analysis
    - Automated alerting and reporting
    - Cloud deployment on Azure
    - Scalable architecture with MongoDB
    """
    
    def __init__(self, config: dict = None):
        """Initialize the anomaly detection system"""
        self.config = config or self._default_config()
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.setup_components()
        
    def _default_config(self) -> dict:
        """Default system configuration"""
        return {
            'batch_size': 1000,
            'model_update_frequency': 24,  # hours
            'anomaly_threshold': 0.05,
            'alert_threshold': 0.8,
            'data_retention_days': 90,
            'enable_real_time': True,
            'enable_alerting': True,
            'models_to_use': ['isolation_forest', 'autoencoder', 'statistical'],
            'database': {
                'mongodb_uri': 'mongodb://localhost:27017/',
                'database_name': 'financial_anomaly_db',
                'collection_name': 'transactions'
            },
            'azure': {
                'storage_account': 'anomalydetection',
                'container_name': 'models',
                'servicebus_namespace': 'anomaly-alerts'
            },
            'monitoring': {
                'enable_wandb': False,
                'slack_webhook': None,
                'email_alerts': False
            }
        }
    
    def setup_components(self):
        """Initialize all system components"""
        try:
            # Setup database connections
            self.setup_database()
            
            # Initialize models
            self.initialize_models()
            
            # Setup monitoring
            self.setup_monitoring()
            
            logger.info("Financial Anomaly Detection system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing system: {str(e)}")
            raise
    
    def setup_database(self):
        """Setup MongoDB connection for transaction storage"""
        try:
            self.mongo_client = pymongo.MongoClient(
                self.config['database']['mongodb_uri']
            )
            self.db = self.mongo_client[self.config['database']['database_name']]
            self.collection = self.db[self.config['database']['collection_name']]
            
            # Create indexes for faster queries
            self.collection.create_index([("timestamp", 1)])
            self.collection.create_index([("user_id", 1)])
            self.collection.create_index([("amount", 1)])
            
            logger.info("Database connection established")
            
        except Exception as e:
            logger.error(f"Database setup failed: {str(e)}")
            # Fallback to local storage
            self.use_local_storage = True
    
    def setup_monitoring(self):
        """Setup monitoring and alerting systems"""
        try:
            # Initialize Redis for caching
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            
            # Setup Slack client if webhook provided
            if self.config['monitoring']['slack_webhook']:
                self.slack_client = slack_sdk.WebhookClient(
                    url=self.config['monitoring']['slack_webhook']
                )
            
            logger.info("Monitoring systems initialized")
            
        except Exception as e:
            logger.warning(f"Monitoring setup failed: {str(e)}")
    
    def initialize_models(self):
        """Initialize anomaly detection models"""
        
        # 1. Isolation Forest for outlier detection
        self.models['isolation_forest'] = IsolationForest(
            contamination=self.config['anomaly_threshold'],
            random_state=42,
            n_estimators=100
        )
        
        # 2. One-Class SVM for novelty detection
        self.models['one_class_svm'] = OneClassSVM(
            gamma='scale',
            nu=self.config['anomaly_threshold']
        )
        
        # 3. DBSCAN for cluster-based anomaly detection
        self.models['dbscan'] = DBSCAN(
            eps=0.5,
            min_samples=5
        )
        
        # 4. Statistical methods
        self.models['statistical'] = StatisticalAnomalyDetector()
        
        # 5. Autoencoder for deep learning-based detection
        self.models['autoencoder'] = None  # Will be built when data is available
        
        # Initialize scalers
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        logger.info("Anomaly detection models initialized")
    
    def generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic financial transaction data for demonstration
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Synthetic transaction data
        """
        np.random.seed(42)
        
        # Generate normal transaction patterns
        normal_data = {
            'user_id': np.random.randint(1000, 9999, n_samples),
            'amount': np.random.lognormal(mean=3, sigma=1, size=n_samples),
            'transaction_type': np.random.choice(
                ['purchase', 'withdrawal', 'transfer', 'deposit'], 
                n_samples, 
                p=[0.5, 0.2, 0.2, 0.1]
            ),
            'merchant_category': np.random.choice(
                ['grocery', 'gas', 'restaurant', 'retail', 'online', 'other'],
                n_samples,
                p=[0.25, 0.15, 0.2, 0.15, 0.15, 0.1]
            ),
            'hour_of_day': np.random.normal(14, 4, n_samples) % 24,
            'day_of_week': np.random.randint(0, 7, n_samples),
            'account_age_days': np.random.exponential(365, n_samples),
            'account_balance': np.random.lognormal(8, 1, n_samples),
            'location_risk_score': np.random.beta(2, 8, n_samples),
            'velocity_1h': np.random.poisson(1, n_samples),
            'velocity_24h': np.random.poisson(5, n_samples)
        }
        
        df = pd.DataFrame(normal_data)
        
        # Add timestamp
        start_date = datetime.now() - timedelta(days=30)
        df['timestamp'] = [
            start_date + timedelta(
                seconds=np.random.randint(0, 30*24*3600)
            ) for _ in range(n_samples)
        ]
        
        # Create derived features
        df['amount_log'] = np.log1p(df['amount'])
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)).astype(int)
        df['balance_ratio'] = df['amount'] / df['account_balance']
        
        # Add anomalies (5% of data)
        n_anomalies = int(n_samples * 0.05)
        anomaly_indices = np.random.choice(df.index, n_anomalies, replace=False)
        
        # Create different types of anomalies
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['amount', 'velocity', 'location', 'time'])
            
            if anomaly_type == 'amount':
                # Unusual large amounts
                df.loc[idx, 'amount'] = np.random.uniform(10000, 50000)
            elif anomaly_type == 'velocity':
                # High transaction velocity
                df.loc[idx, 'velocity_1h'] = np.random.randint(10, 50)
                df.loc[idx, 'velocity_24h'] = np.random.randint(50, 200)
            elif anomaly_type == 'location':
                # High-risk location
                df.loc[idx, 'location_risk_score'] = np.random.uniform(0.8, 1.0)
            elif anomaly_type == 'time':
                # Unusual time patterns
                df.loc[idx, 'hour_of_day'] = np.random.choice([2, 3, 4])
        
        # Add ground truth labels
        df['is_anomaly'] = 0
        df.loc[anomaly_indices, 'is_anomaly'] = 1
        
        # Recalculate derived features for anomalies
        df['amount_log'] = np.log1p(df['amount'])
        df['balance_ratio'] = df['amount'] / df['account_balance']
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess transaction data for anomaly detection
        
        Args:
            df (pd.DataFrame): Raw transaction data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        try:
            # Handle missing values
            df = df.fillna(df.median(numeric_only=True))
            
            # Feature engineering
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Encode categorical variables
            df_encoded = pd.get_dummies(
                df, 
                columns=['transaction_type', 'merchant_category'],
                prefix=['txn', 'merchant']
            )
            
            # Select features for modeling
            feature_cols = [
                'amount_log', 'account_age_days', 'location_risk_score',
                'velocity_1h', 'velocity_24h', 'balance_ratio', 'is_weekend',
                'is_night', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
            ]
            
            # Add encoded categorical features
            encoded_cols = [col for col in df_encoded.columns if col.startswith(('txn_', 'merchant_'))]
            feature_cols.extend(encoded_cols)
            
            self.feature_columns = feature_cols
            
            return df_encoded
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise
    
    def train_models(self, df: pd.DataFrame):
        """
        Train all anomaly detection models
        
        Args:
            df (pd.DataFrame): Training data
        """
        try:
            # Prepare features
            X = df[self.feature_columns]
            
            # Scale features
            X_scaled = self.scalers['standard'].fit_transform(X)
            X_robust = self.scalers['robust'].fit_transform(X)
            
            # Train models
            logger.info("Training Isolation Forest...")
            self.models['isolation_forest'].fit(X_scaled)
            
            logger.info("Training One-Class SVM...")
            self.models['one_class_svm'].fit(X_robust)
            
            logger.info("Training statistical models...")
            self.models['statistical'].fit(df)
            
            # Train autoencoder
            logger.info("Training Autoencoder...")
            self.models['autoencoder'] = self._build_autoencoder(X_scaled.shape[1])
            self.models['autoencoder'].fit(
                X_scaled, X_scaled,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Evaluate models if ground truth is available
            if 'is_anomaly' in df.columns:
                self.evaluate_models(df)
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def _build_autoencoder(self, input_dim: int):
        """Build and compile autoencoder model"""
        
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return autoencoder
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies using ensemble of models
        
        Args:
            df (pd.DataFrame): Transaction data
            
        Returns:
            pd.DataFrame: Data with anomaly scores and predictions
        """
        try:
            X = df[self.feature_columns]
            X_scaled = self.scalers['standard'].transform(X)
            X_robust = self.scalers['robust'].transform(X)
            
            # Get predictions from each model
            predictions = {}
            scores = {}
            
            # Isolation Forest
            predictions['isolation_forest'] = self.models['isolation_forest'].predict(X_scaled)
            scores['isolation_forest'] = self.models['isolation_forest'].decision_function(X_scaled)
            
            # One-Class SVM
            predictions['one_class_svm'] = self.models['one_class_svm'].predict(X_robust)
            scores['one_class_svm'] = self.models['one_class_svm'].decision_function(X_robust)
            
            # Autoencoder
            reconstructed = self.models['autoencoder'].predict(X_scaled)
            reconstruction_error = np.mean(np.square(X_scaled - reconstructed), axis=1)
            scores['autoencoder'] = -reconstruction_error  # Negative for consistency
            
            # Statistical methods
            stat_scores = self.models['statistical'].predict(df)
            scores['statistical'] = stat_scores
            
            # Ensemble scoring
            ensemble_scores = np.mean([
                scores['isolation_forest'],
                scores['one_class_svm'],
                scores['autoencoder'],
                scores['statistical']
            ], axis=0)
            
            # Convert to anomaly probabilities
            anomaly_probabilities = 1 / (1 + np.exp(ensemble_scores))
            
            # Add results to dataframe
            result_df = df.copy()
            result_df['anomaly_score'] = ensemble_scores
            result_df['anomaly_probability'] = anomaly_probabilities
            result_df['is_anomaly_pred'] = (anomaly_probabilities > self.config['anomaly_threshold']).astype(int)
            
            # Add individual model scores
            for model_name, score in scores.items():
                result_df[f'{model_name}_score'] = score
            
            return result_df
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            raise
    
    def evaluate_models(self, df: pd.DataFrame):
        """Evaluate model performance if ground truth is available"""
        
        if 'is_anomaly' not in df.columns:
            logger.warning("No ground truth available for evaluation")
            return
        
        # Get anomaly predictions
        result_df = self.detect_anomalies(df)
        
        y_true = result_df['is_anomaly']
        y_pred = result_df['is_anomaly_pred']
        y_scores = result_df['anomaly_probability']
        
        # Calculate metrics
        auc_score = roc_auc_score(y_true, y_scores)
        
        print("=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)
        print(f"AUC-ROC Score: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        # Log metrics
        logger.info(f"Model evaluation completed. AUC: {auc_score:.4f}")
    
    def real_time_monitoring(self, transaction: dict) -> dict:
        """
        Real-time anomaly detection for individual transactions
        
        Args:
            transaction (dict): Single transaction data
            
        Returns:
            dict: Anomaly detection result
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([transaction])
            df = self.preprocess_data(df)
            
            # Detect anomalies
            result_df = self.detect_anomalies(df)
            
            result = {
                'transaction_id': transaction.get('transaction_id', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'anomaly_score': float(result_df['anomaly_score'].iloc[0]),
                'anomaly_probability': float(result_df['anomaly_probability'].iloc[0]),
                'is_anomaly': bool(result_df['is_anomaly_pred'].iloc[0]),
                'risk_level': self._get_risk_level(result_df['anomaly_probability'].iloc[0])
            }
            
            # Store result
            self._store_result(result)
            
            # Send alerts if necessary
            if result['is_anomaly'] and self.config['enable_alerting']:
                self._send_alert(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Real-time monitoring failed: {str(e)}")
            return {'error': str(e)}
    
    def _get_risk_level(self, probability: float) -> str:
        """Determine risk level based on anomaly probability"""
        if probability < 0.3:
            return 'low'
        elif probability < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def _store_result(self, result: dict):
        """Store anomaly detection result"""
        try:
            # Store in MongoDB
            self.collection.insert_one(result)
            
            # Cache in Redis for quick access
            cache_key = f"anomaly:{result['transaction_id']}"
            self.redis_client.setex(
                cache_key, 
                timedelta(hours=24), 
                json.dumps(result)
            )
            
        except Exception as e:
            logger.error(f"Failed to store result: {str(e)}")
    
    def _send_alert(self, result: dict):
        """Send anomaly alert through configured channels"""
        try:
            message = f"""
            🚨 ANOMALY DETECTED 🚨
            
            Transaction ID: {result['transaction_id']}
            Risk Level: {result['risk_level'].upper()}
            Anomaly Probability: {result['anomaly_probability']:.2%}
            Timestamp: {result['timestamp']}
            
            Please investigate immediately.
            """
            
            # Send Slack alert
            if hasattr(self, 'slack_client'):
                self.slack_client.send(text=message)
            
            # Send email alert (if configured)
            if self.config['monitoring']['email_alerts']:
                self._send_email_alert(message)
            
            logger.info(f"Alert sent for transaction {result['transaction_id']}")
            
        except Exception as e:
            logger.error(f"Failed to send alert: {str(e)}")
    
    def _send_email_alert(self, message: str):
        """Send email alert"""
        # Email implementation would go here
        pass
    
    def generate_dashboard_data(self) -> dict:
        """Generate data for monitoring dashboard"""
        try:
            # Get recent anomalies
            recent_anomalies = list(
                self.collection.find(
                    {'timestamp': {'$gte': datetime.now() - timedelta(hours=24)}}
                ).sort('timestamp', -1).limit(100)
            )
            
            # Calculate metrics
            total_transactions = len(recent_anomalies)
            anomaly_count = sum(1 for a in recent_anomalies if a.get('is_anomaly', False))
            anomaly_rate = (anomaly_count / total_transactions) if total_transactions > 0 else 0
            
            # Risk distribution
            risk_distribution = {
                'low': sum(1 for a in recent_anomalies if a.get('risk_level') == 'low'),
                'medium': sum(1 for a in recent_anomalies if a.get('risk_level') == 'medium'),
                'high': sum(1 for a in recent_anomalies if a.get('risk_level') == 'high')
            }
            
            return {
                'summary': {
                    'total_transactions_24h': total_transactions,
                    'anomalies_detected': anomaly_count,
                    'anomaly_rate': f"{anomaly_rate:.2%}",
                    'system_status': 'operational'
                },
                'risk_distribution': risk_distribution,
                'recent_anomalies': recent_anomalies[:10],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Dashboard data generation failed: {str(e)}")
            return {'error': str(e)}
    
    def save_models(self, model_path: str = 'models/'):
        """Save trained models to disk"""
        try:
            import os
            os.makedirs(model_path, exist_ok=True)
            
            # Save sklearn models
            with open(f'{model_path}/isolation_forest.pkl', 'wb') as f:
                pickle.dump(self.models['isolation_forest'], f)
            
            with open(f'{model_path}/one_class_svm.pkl', 'wb') as f:
                pickle.dump(self.models['one_class_svm'], f)
            
            # Save scalers
            with open(f'{model_path}/scalers.pkl', 'wb') as f:
                pickle.dump(self.scalers, f)
            
            # Save autoencoder
            if self.models['autoencoder']:
                self.models['autoencoder'].save(f'{model_path}/autoencoder.h5')
            
            # Save feature columns
            with open(f'{model_path}/feature_columns.pkl', 'wb') as f:
                pickle.dump(self.feature_columns, f)
            
            logger.info(f"Models saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
    
    def load_models(self, model_path: str = 'models/'):
        """Load trained models from disk"""
        try:
            # Load sklearn models
            with open(f'{model_path}/isolation_forest.pkl', 'rb') as f:
                self.models['isolation_forest'] = pickle.load(f)
            
            with open(f'{model_path}/one_class_svm.pkl', 'rb') as f:
                self.models['one_class_svm'] = pickle.load(f)
            
            # Load scalers
            with open(f'{model_path}/scalers.pkl', 'rb') as f:
                self.scalers = pickle.load(f)
            
            # Load autoencoder
            if os.path.exists(f'{model_path}/autoencoder.h5'):
                self.models['autoencoder'] = tf.keras.models.load_model(f'{model_path}/autoencoder.h5')
            
            # Load feature columns
            with open(f'{model_path}/feature_columns.pkl', 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            logger.info(f"Models loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")

class StatisticalAnomalyDetector:
    """Statistical anomaly detection using z-scores and percentiles"""
    
    def __init__(self):
        self.stats = {}
    
    def fit(self, df: pd.DataFrame):
        """Fit statistical parameters"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'q95': df[col].quantile(0.95),
                'q05': df[col].quantile(0.05)
            }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict anomaly scores using statistical methods"""
        scores = []
        
        for _, row in df.iterrows():
            row_score = 0
            count = 0
            
            for col, stats in self.stats.items():
                if col in row.index:
                    value = row[col]
                    z_score = abs((value - stats['mean']) / (stats['std'] + 1e-8))
                    
                    # Check if value is in extreme percentiles
                    if value > stats['q95'] or value < stats['q05']:
                        z_score += 2
                    
                    row_score += z_score
                    count += 1
            
            avg_score = row_score / (count + 1e-8)
            scores.append(-avg_score)  # Negative for consistency with other models
        
        return np.array(scores)

def create_flask_app(detector: FinancialAnomalyDetector) -> Flask:
    """Create Flask API for the anomaly detection system"""
    
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
    
    @app.route('/detect', methods=['POST'])
    def detect_anomaly():
        """Real-time anomaly detection endpoint"""
        try:
            transaction = request.json
            result = detector.real_time_monitoring(transaction)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/dashboard', methods=['GET'])
    def dashboard():
        """Dashboard data endpoint"""
        try:
            data = detector.generate_dashboard_data()
            return jsonify(data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/retrain', methods=['POST'])
    def retrain_models():
        """Model retraining endpoint"""
        try:
            # This would trigger model retraining in production
            return jsonify({'status': 'retraining initiated'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

def main():
    """
    Main execution function demonstrating the Financial Anomaly Detection system
    """
    print("🔍 Financial Anomaly Detection System")
    print("=" * 50)
    
    # Initialize the system
    config = {
        'anomaly_threshold': 0.05,
        'enable_real_time': True,
        'enable_alerting': False,  # Disabled for demo
        'monitoring': {
            'enable_wandb': False,
            'slack_webhook': None,
            'email_alerts': False
        }
    }
    
    try:
        detector = FinancialAnomalyDetector(config)
        
        print("\n📊 Generating synthetic financial transaction data...")
        df = detector.generate_synthetic_data(n_samples=5000)
        print(f"Generated {len(df)} transactions with {df['is_anomaly'].sum()} anomalies")
        
        print("\n🔧 Preprocessing data...")
        df_processed = detector.preprocess_data(df)
        print(f"Processed data shape: {df_processed.shape}")
        
        print("\n🤖
