"""
Customer Segmentation for Targeted Marketing
Advanced segmentation using RFM analysis and machine learning clustering techniques

Author: Durga Katreddi
Email: katreddisrisaidurga@gmail.com
LinkedIn: https://linkedin.com/in/sri-sai-durga-katreddi-
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Clustering and Analysis
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.mixture import GaussianMixture

# Data Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Business Intelligence
import sqlite3
from sqlalchemy import create_engine

# Campaign Integration
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import json

# Monitoring and Logging
import logging
from datetime import datetime
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomerSegmentation:
    """
    Advanced Customer Segmentation System for Targeted Marketing
    
    Features:
    - RFM (Recency, Frequency, Monetary) Analysis
    - Advanced clustering algorithms (K-Means, DBSCAN, Hierarchical)
    - Customer lifetime value prediction
    - Behavioral segmentation
    - Campaign optimization and automation
    - ROI tracking and performance analytics
    """
    
    def __init__(self, config: dict = None):
        """Initialize the customer segmentation system"""
        self.config = config or self._default_config()
        self.segments = {}
        self.models = {}
        self.scalers = {}
        self.setup_components()
        
    def _default_config(self) -> dict:
        """Default system configuration"""
        return {
            'segmentation_method': 'rfm_kmeans',
            'n_clusters': 5,
            'rfm_weights': {'recency': 0.3, 'frequency': 0.4, 'monetary': 0.3},
            'campaign_settings': {
                'email_template_path': 'templates/',
                'api_endpoints': {
                    'email_service': 'https://api.emailservice.com/',
                    'crm_system': 'https://api.crm.com/',
                    'analytics': 'https://api.analytics.com/'
                }
            },
            'business_rules': {
                'min_purchase_amount': 10,
                'analysis_period_days': 365,
                'high_value_threshold': 1000,
                'churn_risk_days': 90
            },
            'visualization': {
                'save_plots': True,
                'plot_format': 'png',
                'interactive_plots': True
            }
        }
    
    def setup_components(self):
        """Initialize system components"""
        try:
            # Initialize scalers for different use cases
            self.scalers['standard'] = StandardScaler()
            self.scalers['minmax'] = MinMaxScaler()
            self.scalers['robust'] = RobustScaler()
            
            # Initialize clustering models
            self.models['kmeans'] = KMeans(
                n_clusters=self.config['n_clusters'], 
                random_state=42,
                n_init=10
            )
            
            self.models['dbscan'] = DBSCAN(
                eps=0.5, 
                min_samples=5
            )
            
            self.models['hierarchical'] = AgglomerativeClustering(
                n_clusters=self.config['n_clusters']
            )
            
            self.models['gaussian_mixture'] = GaussianMixture(
                n_components=self.config['n_clusters'],
                random_state=42
            )
            
            logger.info("Customer segmentation system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing system: {str(e)}")
            raise
    
    def generate_synthetic_data(self, n_customers: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic customer transaction data
        
        Args:
            n_customers (int): Number of customers to generate
            
        Returns:
            pd.DataFrame: Synthetic customer transaction data
        """
        np.random.seed(42)
        
        # Generate customer base
        customers = []
        for customer_id in range(1, n_customers + 1):
            # Customer demographics
            age = np.random.normal(40, 15)
            age = max(18, min(80, age))  # Keep age between 18-80
            
            income = np.random.lognormal(mean=10.5, sigma=0.5)
            
            # Customer behavior patterns
            if np.random.random() < 0.1:  # 10% VIP customers
                segment_type = 'VIP'
                purchase_frequency = np.random.poisson(15) + 5  # 5-20+ purchases
                avg_order_value = np.random.normal(500, 200)
            elif np.random.random() < 0.2:  # 20% regular customers
                segment_type = 'Regular'
                purchase_frequency = np.random.poisson(8) + 2  # 2-15+ purchases
                avg_order_value = np.random.normal(150, 50)
            elif np.random.random() < 0.3:  # 30% occasional customers
                segment_type = 'Occasional'
                purchase_frequency = np.random.poisson(3) + 1  # 1-8+ purchases
                avg_order_value = np.random.normal(75, 25)
            else:  # 40% one-time customers
                segment_type = 'One-time'
                purchase_frequency = 1
                avg_order_value = np.random.normal(50, 20)
            
            # Generate transactions for this customer
            last_purchase_days_ago = np.random.exponential(30)
            if segment_type == 'VIP':
                last_purchase_days_ago = min(last_purchase_days_ago, 15)  # VIPs purchase recently
            elif segment_type == 'One-time':
                last_purchase_days_ago = np.random.uniform(60, 365)  # One-time customers haven't purchased recently
            
            total_spent = purchase_frequency * max(10, avg_order_value + np.random.normal(0, avg_order_value * 0.2))
            
            customers.append({
                'customer_id': customer_id,
                'age': int(age),
                'income': max(20000, income),
                'segment_type': segment_type,
                'purchase_frequency': purchase_frequency,
                'avg_order_value': max(10, avg_order_value),
                'total_spent': max(10, total_spent),
                'last_purchase_days_ago': int(last_purchase_days_ago),
                'signup_date': datetime.now() - timedelta(days=np.random.randint(30, 730)),
                'preferred_category': np.random.choice([
                    'Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Beauty'
                ]),
                'marketing_channel': np.random.choice([
                    'Email', 'Social Media', 'Search', 'Direct', 'Referral'
                ], p=[0.3, 0.25, 0.2, 0.15, 0.1]),
                'email_opens': np.random.poisson(purchase_frequency * 2),
                'email_clicks': np.random.poisson(purchase_frequency),
                'website_visits': np.random.poisson(purchase_frequency * 3),
                'mobile_app_usage': np.random.choice([0, 1], p=[0.4, 0.6]),
                'location': np.random.choice([
                    'Urban', 'Suburban', 'Rural'
                ], p=[0.5, 0.35, 0.15])
            })
        
        df = pd.DataFrame(customers)
        
        # Calculate additional features
        df['customer_lifetime_days'] = (datetime.now() - df['signup_date']).dt.days
        df['purchase_rate'] = df['purchase_frequency'] / (df['customer_lifetime_days'] / 30)  # purchases per month
        df['email_engagement_rate'] = df['email_clicks'] / (df['email_opens'] + 1)
        df['recency_score'] = self._calculate_recency_score(df['last_purchase_days_ago'])
        df['frequency_score'] = self._calculate_frequency_score(df['purchase_frequency'])
        df['monetary_score'] = self._calculate_monetary_score(df['total_spent'])
        
        # Add some seasonal and behavioral patterns
        df['is_holiday_shopper'] = (df['purchase_frequency'] > df['purchase_frequency'].quantile(0.7)).astype(int)
        df['is_deal_seeker'] = (df['avg_order_value'] < df['avg_order_value'].quantile(0.3)).astype(int)
        df['is_brand_loyal'] = (df['customer_lifetime_days'] > 365).astype(int)
        
        return df
    
    def _calculate_recency_score(self, recency_days: pd.Series) -> pd.Series:
        """Calculate recency score (1-5, where 5 is most recent)"""
        return pd.cut(recency_days, 
                     bins=[0, 30, 60, 90, 180, float('inf')], 
                     labels=[5, 4, 3, 2, 1], 
                     include_lowest=True).astype(int)
    
    def _calculate_frequency_score(self, frequency: pd.Series) -> pd.Series:
        """Calculate frequency score (1-5, where 5 is most frequent)"""
        return pd.cut(frequency, 
                     bins=[0, 1, 3, 6, 10, float('inf')], 
                     labels=[1, 2, 3, 4, 5], 
                     include_lowest=True).astype(int)
    
    def _calculate_monetary_score(self, monetary: pd.Series) -> pd.Series:
        """Calculate monetary score (1-5, where 5 is highest value)"""
        return pd.cut(monetary, 
                     bins=[0, 100, 300, 600, 1200, float('inf')], 
                     labels=[1, 2, 3, 4, 5], 
                     include_lowest=True).astype(int)
    
    def perform_rfm_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform comprehensive RFM analysis
        
        Args:
            df (pd.DataFrame): Customer data
            
        Returns:
            pd.DataFrame: Data with RFM scores and segments
        """
        try:
            logger.info("Performing RFM analysis...")
            
            # Create RFM dataframe
            rfm_df = df.copy()
            
            # Calculate RFM scores (already done in synthetic data, but we'll recalculate for real data)
            rfm_df['R'] = rfm_df['recency_score']
            rfm_df['F'] = rfm_df['frequency_score'] 
            rfm_df['M'] = rfm_df['monetary_score']
            
            # Calculate composite RFM score
            weights = self.config['rfm_weights']
            rfm_df['RFM_Score'] = (
                rfm_df['R'] * weights['recency'] + 
                rfm_df['F'] * weights['frequency'] + 
                rfm_df['M'] * weights['monetary']
            )
            
            # Create RFM segments based on scores
            rfm_df['RFM_Segment'] = rfm_df.apply(self._assign_rfm_segment, axis=1)
            
            # Calculate customer lifetime value
            rfm_df['CLV'] = self._calculate_clv(rfm_df)
            
            logger.info(f"RFM analysis completed for {len(rfm_df)} customers")
            return rfm_df
            
        except Exception as e:
            logger.error(f"RFM analysis failed: {str(e)}")
            raise
    
    def _assign_rfm_segment(self, row: pd.Series) -> str:
        """Assign RFM segment based on R, F, M scores"""
        R, F, M = row['R'], row['F'], row['M']
        
        if R >= 4 and F >= 4 and M >= 4:
            return 'Champions'
        elif R >= 3 and F >= 3 and M >= 3:
            return 'Loyal Customers'
        elif R >= 4 and F >= 2:
            return 'Potential Loyalists'
        elif R >= 4 and F == 1:
            return 'New Customers'
        elif R >= 3 and F >= 2 and M >= 2:
            return 'Promising'
        elif R >= 2 and F >= 2 and M >= 2:
            return 'Customers Needing Attention'
        elif R >= 2 and F >= 3:
            return 'About to Sleep'
        elif R <= 2 and F >= 2:
            return 'At Risk'
        elif R <= 2 and F >= 4 and M >= 4:
            return 'Cannot Lose Them'
        elif R <= 1:
            return 'Lost'
        else:
            return 'Others'
    
    def _calculate_clv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Customer Lifetime Value"""
        # Simple CLV calculation: (Average Order Value * Purchase Frequency * Gross Margin * Lifespan)
        # Assuming 20% gross margin and average lifespan based on recency
        gross_margin = 0.2
        avg_lifespan_months = 24  # 2 years average
        
        clv = (
            df['avg_order_value'] * 
            df['purchase_frequency'] * 
            gross_margin * 
            avg_lifespan_months
        )
        
        return clv
    
    def perform_advanced_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform advanced clustering analysis using multiple algorithms
        
        Args:
            df (pd.DataFrame): Customer data with RFM scores
            
        Returns:
            pd.DataFrame: Data with cluster assignments
        """
        try:
            logger.info("Performing advanced clustering analysis...")
            
            # Select features for clustering
            clustering_features = [
                'R', 'F', 'M', 'age', 'income', 'email_engagement_rate',
                'purchase_rate', 'website_visits', 'is_holiday_shopper',
                'is_deal_seeker', 'is_brand_loyal'
            ]
            
            X = df[clustering_features].copy()
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Scale features
            X_scaled = self.scalers['standard'].fit_transform(X)
            
            # Apply different clustering algorithms
            results_df = df.copy()
            
            # K-Means clustering
            kmeans_labels = self.models['kmeans'].fit_predict(X_scaled)
            results_df['KMeans_Cluster'] = kmeans_labels
            
            # DBSCAN clustering
            dbscan_labels = self.models['dbscan'].fit_predict(X_scaled)
            results_df['DBSCAN_Cluster'] = dbscan_labels
            
            # Hierarchical clustering
            hierarchical_labels = self.models['hierarchical'].fit_predict(X_scaled)
            results_df['Hierarchical_Cluster'] = hierarchical_labels
            
            # Gaussian Mixture clustering
            gmm_labels = self.models['gaussian_mixture'].fit_predict(X_scaled)
            results_df['GMM_Cluster'] = gmm_labels
            
            # Evaluate clustering quality
            self._evaluate_clustering_quality(X_scaled, results_df)
            
            # Select best clustering method (using silhouette score)
            best_method = self._select_best_clustering_method(X_scaled, results_df)
            results_df['Best_Cluster'] = results_df[f'{best_method}_Cluster']
            
            # Create comprehensive segment profiles
            results_df = self._create_segment_profiles(results_df)
            
            logger.info("Advanced clustering analysis completed")
            return results_df
            
        except Exception as e:
            logger.error(f"Advanced clustering failed: {str(e)}")
            raise
    
    def _evaluate_clustering_quality(self, X_scaled: np.ndarray, results_df: pd.DataFrame):
        """Evaluate the quality of different clustering methods"""
        methods = ['KMeans', 'DBSCAN', 'Hierarchical', 'GMM']
        evaluation_results = {}
        
        for method in methods:
            labels = results_df[f'{method}_Cluster']
            
            # Skip evaluation if only one cluster or too many noise points
            if len(set(labels)) <= 1 or (method == 'DBSCAN' and list(labels).count(-1) > len(labels) * 0.5):
                evaluation_results[method] = {'silhouette_score': -1, 'calinski_harabasz_score': 0}
                continue
            
            try:
                silhouette_avg = silhouette_score(X_scaled, labels)
                calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
                
                evaluation_results[method] = {
                    'silhouette_score': silhouette_avg,
                    'calinski_harabasz_score': calinski_harabasz
                }
            except:
                evaluation_results[method] = {'silhouette_score': -1, 'calinski_harabasz_score': 0}
        
        self.clustering_evaluation = evaluation_results
        
        print("\n" + "="*60)
        print("CLUSTERING EVALUATION RESULTS")
        print("="*60)
        for method, scores in evaluation_results.items():
            print(f"{method:15} | Silhouette: {scores['silhouette_score']:.4f} | Calinski-Harabasz: {scores['calinski_harabasz_score']:.2f}")
    
    def _select_best_clustering_method(self, X_scaled: np.ndarray, results_df: pd.DataFrame) -> str:
        """Select the best clustering method based on evaluation metrics"""
        best_method = 'KMeans'  # Default
        best_score = -1
        
        for method, scores in self.clustering_evaluation.items():
            if scores['silhouette_score'] > best_score:
                best_score = scores['silhouette_score']
                best_method = method
        
        logger.info(f"Best clustering method selected: {best_method} (Silhouette Score: {best_score:.4f})")
        return best_method
    
    def _create_segment_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create detailed profiles for each customer segment"""
        
        # Use the best clustering method
        segment_col = 'Best_Cluster'
        
        segment_profiles = {}
        
        for segment in df[segment_col].unique():
            segment_data = df[df[segment_col] == segment]
            
            profile = {
                'size': len(segment_data),
                'percentage': len(segment_data) / len(df) * 100,
                'avg_clv': segment_data['CLV'].mean(),
                'avg_total_spent': segment_data['total_spent'].mean(),
                'avg_purchase_frequency': segment_data['purchase_frequency'].mean(),
                'avg_recency': segment_data['last_purchase_days_ago'].mean(),
                'avg_age': segment_data['age'].mean(),
                'avg_income': segment_data['income'].mean(),
                'top_category': segment_data['preferred_category'].mode().iloc[0] if not segment_data['preferred_category'].mode().empty else 'Unknown',
                'top_channel': segment_data['marketing_channel'].mode().iloc[0] if not segment_data['marketing_channel'].mode().empty else 'Unknown',
                'email_engagement': segment_data['email_engagement_rate'].mean(),
                'mobile_usage_rate': segment_data['mobile_app_usage'].mean()
            }
            
            segment_profiles[segment] = profile
        
        self.segment_profiles = segment_profiles
        return df
    
    def generate_campaign_recommendations(self, df: pd.DataFrame) -> dict:
        """
        Generate targeted marketing campaign recommendations for each segment
        
        Args:
            df (pd.DataFrame): Customer data with segments
            
        Returns:
            dict: Campaign recommendations by segment
        """
        try:
            campaigns = {}
            
            for segment in df['Best_Cluster'].unique():
                segment_data = df[df['Best_Cluster'] == segment]
                profile = self.segment_profiles[segment]
                
                # Generate campaign strategy based on segment characteristics
                if profile['avg_clv'] > df['CLV'].quantile(0.8):
                    # High-value segment
                    campaigns[segment] = {
                        'campaign_type': 'VIP Program',
                        'channel': 'Email + Personal Outreach',
                        'message': 'Exclusive premium offers and early access',
                        'frequency': 'Weekly',
                        'budget_allocation': 0.3,
                        'expected_conversion': 0.15,
                        'tactics': [
                            'Personalized product recommendations',
                            'VIP customer service',
                            'Exclusive events and previews',
                            'Loyalty program benefits'
                        ]
                    }
                elif profile['avg_recency'] > 60:
                    # At-risk or dormant segment
                    campaigns[segment] = {
                        'campaign_type': 'Win-Back Campaign',
                        'channel': 'Email + Retargeting Ads',
                        'message': 'We miss you! Special comeback offers',
                        'frequency': 'Bi-weekly',
                        'budget_allocation': 0.2,
                        'expected_conversion': 0.08,
                        'tactics': [
                            'Discount offers (15-25% off)',
                            'Free shipping incentives',
                            'Product recommendations based on past purchases',
                            'Survey to understand absence reasons'
                        ]
                    }
                elif profile['avg_purchase_frequency'] < 2:
                    # Low-frequency segment
                    campaigns[segment] = {
                        'campaign_type': 'Engagement Building',
                        'channel': 'Social Media + Content Marketing',
                        'message': 'Discover what you\'ve been missing',
                        'frequency': 'Monthly',
                        'budget_allocation': 0.15,
                        'expected_conversion': 0.05,
                        'tactics': [
                            'Educational content',
                            'Product tutorials and tips',
                            'User-generated content campaigns',
                            'Small incentives for engagement'
                        ]
                    }
                else:
                    # Regular customers
                    campaigns[segment] = {
                        'campaign_type': 'Loyalty Enhancement',
                        'channel': 'Email + Mobile App',
                        'message': 'Thank you for being a valued customer',
                        'frequency': 'Bi-weekly',
                        'budget_allocation': 0.25,
                        'expected_conversion': 0.12,
                        'tactics': [
                            'Cross-sell related products',
                            'Seasonal promotions',
                            'Referral incentives',
                            'Birthday and anniversary offers'
                        ]
                    }
                
                # Add segment-specific metrics
                campaigns[segment]['segment_size'] = profile['size']
                campaigns[segment]['avg_order_value'] = profile['avg_total_spent'] / max(1, profile['avg_purchase_frequency'])
                campaigns[segment]['potential_revenue'] = (
                    profile['size'] * 
                    campaigns[segment]['expected_conversion'] * 
                    campaigns[segment]['avg_order_value']
                )
            
            logger.info(f"Generated campaign recommendations for {len(campaigns)} segments")
            return campaigns
            
        except Exception as e:
            logger.error(f"Campaign recommendation generation failed: {str(e)}")
            raise
    
    def simulate_campaign_performance(self, campaigns: dict, df: pd.DataFrame) -> dict:
        """
        Simulate campaign performance and ROI
        
        Args:
            campaigns (dict): Campaign recommendations
            df (pd.DataFrame): Customer data
            
        Returns:
            dict: Simulated performance metrics
        """
        try:
            performance_results = {}
            total_budget = 100000  # $100k total marketing budget
            
            for segment, campaign in campaigns.items():
                segment_data = df[df['Best_Cluster'] == segment]
                
                # Calculate budget allocation
                allocated_budget = total_budget * campaign['budget_allocation']
                
                # Simulate campaign performance
                targeted_customers = len(segment_data)
                expected_conversions = int(targeted_customers * campaign['expected_conversion'])
                avg_order_value = campaign['avg_order_value']
                
                # Revenue calculations
                gross_revenue = expected_conversions * avg_order_value
                net_revenue = gross_revenue - allocated_budget
                roi = (net_revenue / allocated_budget) * 100 if allocated_budget > 0 else 0
                
                # Performance metrics
                performance_results[segment] = {
                    'campaign_type': campaign['campaign_type'],
                    'allocated_budget': allocated_budget,
                    'targeted_customers': targeted_customers,
                    'expected_conversions': expected_conversions,
                    'conversion_rate': campaign['expected_conversion'],
                    'gross_revenue': gross_revenue,
                    'net_revenue': net_revenue,
                    'roi_percentage': roi,
                    'cost_per_acquisition': allocated_budget / max(1, expected_conversions),
                    'customer_lifetime_value': segment_data['CLV'].mean()
                }
            
            # Calculate overall campaign performance
            total_allocated_budget = sum(r['allocated_budget'] for r in performance_results.values())
            total_gross_revenue = sum(r['gross_revenue'] for r in performance_results.values())
            total_net_revenue = sum(r['net_revenue'] for r in performance_results.values())
            overall_roi = (total_net_revenue / total_allocated_budget) * 100
            
            performance_results['OVERALL'] = {
                'total_budget': total_allocated_budget,
                'total_gross_revenue': total_gross_revenue,
                'total_net_revenue': total_net_revenue,
                'overall_roi': overall_roi,
                'total_conversions': sum(r['expected_conversions'] for r in performance_results.values() if isinstance(r, dict) and 'expected_conversions' in r)
            }
            
            logger.info(f"Campaign performance simulation completed. Overall ROI: {overall_roi:.1f}%")
            return performance_results
            
        except Exception as e:
            logger.error(f"Campaign performance simulation failed: {str(e)}")
            raise
    
    def create_visualizations(self, df: pd.DataFrame) -> dict:
        """
        Create comprehensive visualizations for customer segmentation analysis
        
        Args:
            df (pd.DataFrame): Customer data with segments
            
        Returns:
            dict: Dictionary of plot objects
        """
        try:
            plots = {}
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. RFM Segment Distribution
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            segment_counts = df['RFM_Segment'].value_counts()
            sns.barplot(x=segment_counts.values, y=segment_counts.index, ax=ax)
            ax.set_title('Customer Distribution by RFM Segments', fontsize=16, fontweight='bold')
            ax.set_xlabel('Number of Customers')
            plt.tight_layout()
            plots['rfm_distribution'] = fig
            
            # 2. CLV by Segment
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            sns.boxplot(data=df, x='Best_Cluster', y='CLV', ax=ax)
            ax.set_title('Customer Lifetime Value by Cluster', fontsize=16, fontweight='bold')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Customer Lifetime Value ($)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plots['clv_by_segment'] = fig
            
            # 3. RFM Heatmap
            rfm_summary = df.groupby('RFM_Segment').agg({
                'R': 'mean',
                'F': 'mean', 
                'M': 'mean',
                'customer_id': 'count'
            }).round(2)
            rfm_summary.columns = ['Recency', 'Frequency', 'Monetary', 'Count']
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            sns.heatmap(rfm_summary[['Recency', 'Frequency', 'Monetary']], 
                       annot=True, cmap='RdYlBu_r', ax=ax)
            ax.set_title('RFM Scores by Segment', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plots['rfm_heatmap'] = fig
            
            # 4. Cluster Scatter Plot (PCA)
            clustering_features = ['R', 'F', 'M', 'age', 'income', 'email_engagement_rate']
            X = df[clustering_features].fillna(df[clustering_features].median())
            X_scaled = self.scalers['standard'].transform(X)
            
            pca = PCA(n_components=2
