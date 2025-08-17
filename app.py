import os
import json
import traceback
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO, BytesIO
import base64
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import sqlite3
import tempfile
import re
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yfinance as yf
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

class DataAnalystAgent:
    def __init__(self):
        self.data_cache = {}
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json', '.txt']
        
    def load_data_from_file(self, file_path: str, filename: str) -> Optional[pd.DataFrame]:
        """Load data from various file formats"""
        try:
            file_ext = os.path.splitext(filename.lower())[1]
            
            if file_ext == '.csv':
                # Try different encodings and separators
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    for sep in [',', ';', '\t', '|']:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                            if df.shape[1] > 1:  # More than one column suggests correct separator
                                return df
                        except:
                            continue
                # Fallback
                return pd.read_csv(file_path)
                
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
                
            elif file_ext == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    return pd.DataFrame([data])
                    
            elif file_ext == '.txt':
                # Try to parse as CSV first
                try:
                    df = pd.read_csv(file_path, sep=None, engine='python')
                    return df
                except:
                    # If that fails, read as plain text
                    with open(file_path, 'r') as f:
                        content = f.read()
                    return pd.DataFrame({'text': [content]})
                    
        except Exception as e:
            logger.error(f"Error loading file {filename}: {str(e)}")
            return None
        
        return None
    
    def fetch_financial_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch financial data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            logger.error(f"Error fetching financial data for {symbol}: {str(e)}")
            return None
    
    def fetch_public_data(self, data_source: str) -> Optional[pd.DataFrame]:
        """Fetch data from public APIs or datasets"""
        try:
            # Example implementations for common data sources
            if "covid" in data_source.lower():
                url = "https://disease.sh/v3/covid-19/countries"
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    return pd.DataFrame(response.json())
                    
            elif "weather" in data_source.lower():
                # This would require an API key in real implementation
                logger.warning("Weather data requires API key")
                return None
                
            # Add more data sources as needed
            
        except Exception as e:
            logger.error(f"Error fetching public data: {str(e)}")
            return None
        
        return None
    
    def analyze_data_basic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform basic data analysis"""
        try:
            analysis = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
            
            # Numeric columns analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                analysis['numeric_summary'] = df[numeric_cols].describe().to_dict()
                
            # Categorical columns analysis
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                analysis['categorical_summary'] = {}
                for col in cat_cols[:5]:  # Limit to first 5 categorical columns
                    analysis['categorical_summary'][col] = df[col].value_counts().head().to_dict()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in basic analysis: {str(e)}")
            return {'error': str(e)}
    
    def perform_statistical_analysis(self, df: pd.DataFrame, analysis_type: str) -> Dict[str, Any]:
        """Perform statistical analysis"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if analysis_type.lower() in ['correlation', 'corr']:
                if len(numeric_cols) >= 2:
                    corr_matrix = df[numeric_cols].corr()
                    return {'correlation_matrix': corr_matrix.to_dict()}
                    
            elif analysis_type.lower() in ['regression', 'linear']:
                if len(numeric_cols) >= 2:
                    target_col = numeric_cols[-1]  # Use last column as target
                    feature_cols = numeric_cols[:-1]
                    
                    X = df[feature_cols].dropna()
                    y = df[target_col].loc[X.index]
                    
                    if len(X) > 10:  # Need enough data points
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = model.score(X_test, y_test)
                        
                        return {
                            'model': 'linear_regression',
                            'mse': float(mse),
                            'r2_score': float(r2),
                            'coefficients': dict(zip(feature_cols, model.coef_)),
                            'intercept': float(model.intercept_)
                        }
                        
            elif analysis_type.lower() in ['ttest', 't-test']:
                if len(numeric_cols) >= 1:
                    col = numeric_cols[0]
                    data = df[col].dropna()
                    if len(data) > 5:
                        t_stat, p_value = stats.ttest_1samp(data, data.mean())
                        return {
                            'test': 't-test',
                            'statistic': float(t_stat),
                            'p_value': float(p_value),
                            'mean': float(data.mean()),
                            'std': float(data.std())
                        }
            
            return {'error': 'Analysis type not supported or insufficient data'}
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {str(e)}")
            return {'error': str(e)}
    
    def create_visualization(self, df: pd.DataFrame, viz_type: str, **kwargs) -> Optional[str]:
        """Create visualization and return base64 encoded image"""
        try:
            plt.figure(figsize=(10, 6))
            plt.style.use('default')
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if viz_type.lower() in ['histogram', 'hist']:
                if len(numeric_cols) > 0:
                    col = kwargs.get('column', numeric_cols[0])
                    if col in df.columns:
                        plt.hist(df[col].dropna(), bins=30, alpha=0.7)
                        plt.title(f'Histogram of {col}')
                        plt.xlabel(col)
                        plt.ylabel('Frequency')
                        
            elif viz_type.lower() in ['scatter', 'scatterplot']:
                if len(numeric_cols) >= 2:
                    x_col = kwargs.get('x', numeric_cols[0])
                    y_col = kwargs.get('y', numeric_cols[1])
                    if x_col in df.columns and y_col in df.columns:
                        plt.scatter(df[x_col], df[y_col], alpha=0.6)
                        plt.title(f'{y_col} vs {x_col}')
                        plt.xlabel(x_col)
                        plt.ylabel(y_col)
                        
            elif viz_type.lower() in ['line', 'lineplot', 'time_series']:
                if len(numeric_cols) > 0:
                    col = kwargs.get('column', numeric_cols[0])
                    if col in df.columns:
                        plt.plot(df.index, df[col])
                        plt.title(f'Line Plot of {col}')
                        plt.xlabel('Index')
                        plt.ylabel(col)
                        
            elif viz_type.lower() in ['bar', 'barplot']:
                cat_cols = df.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    col = kwargs.get('column', cat_cols[0])
                    if col in df.columns:
                        value_counts = df[col].value_counts().head(10)
                        plt.bar(range(len(value_counts)), value_counts.values)
                        plt.title(f'Bar Plot of {col}')
                        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
                        plt.ylabel('Count')
                        
            elif viz_type.lower() in ['heatmap', 'correlation']:
                if len(numeric_cols) >= 2:
                    corr_matrix = df[numeric_cols].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                    plt.title('Correlation Heatmap')
                    
            plt.tight_layout()
            
            # Save plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_data
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            plt.close()
            return None
    
    def answer_question(self, question: str, available_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Answer a specific question using available data"""
        try:
            question_lower = question.lower()
            
            # If no data available, try to infer what data is needed
            if not available_data:
                if any(keyword in question_lower for keyword in ['stock', 'price', 'financial', 'ticker']):
                    # Try to extract stock symbol
                    words = question.split()
                    for word in words:
                        if word.isupper() and len(word) <= 5:
                            df = self.fetch_financial_data(word)
                            if df is not None:
                                available_data[f'{word}_stock'] = df
                                break
                
                elif any(keyword in question_lower for keyword in ['covid', 'coronavirus', 'pandemic']):
                    df = self.fetch_public_data('covid')
                    if df is not None:
                        available_data['covid_data'] = df
            
            # If still no data, return error
            if not available_data:
                return {
                    'answer': 'No data available to answer this question.',
                    'question': question,
                    'status': 'no_data'
                }
            
            # Use the first available dataset for analysis
            df_name, df = next(iter(available_data.items()))
            
            # Analyze question type and provide appropriate answer
            if any(keyword in question_lower for keyword in ['correlation', 'correlate', 'relationship']):
                result = self.perform_statistical_analysis(df, 'correlation')
                viz = self.create_visualization(df, 'heatmap')
                return {
                    'answer': result,
                    'question': question,
                    'visualization': viz,
                    'data_source': df_name,
                    'status': 'success'
                }
                
            elif any(keyword in question_lower for keyword in ['trend', 'over time', 'time series']):
                viz = self.create_visualization(df, 'line')
                basic_stats = self.analyze_data_basic(df)
                return {
                    'answer': basic_stats,
                    'question': question,
                    'visualization': viz,
                    'data_source': df_name,
                    'status': 'success'
                }
                
            elif any(keyword in question_lower for keyword in ['distribution', 'histogram']):
                viz = self.create_visualization(df, 'histogram')
                basic_stats = self.analyze_data_basic(df)
                return {
                    'answer': basic_stats,
                    'question': question,
                    'visualization': viz,
                    'data_source': df_name,
                    'status': 'success'
                }
                
            elif any(keyword in question_lower for keyword in ['predict', 'regression', 'forecast']):
                result = self.perform_statistical_analysis(df, 'regression')
                viz = self.create_visualization(df, 'scatter')
                return {
                    'answer': result,
                    'question': question,
                    'visualization': viz,
                    'data_source': df_name,
                    'status': 'success'
                }
            
            # Default: provide basic analysis
            basic_stats = self.analyze_data_basic(df)
            viz = self.create_visualization(df, 'histogram')
            
            return {
                'answer': basic_stats,
                'question': question,
                'visualization': viz,
                'data_source': df_name,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error answering question '{question}': {str(e)}")
            return {
                'answer': f'Error processing question: {str(e)}',
                'question': question,
                'status': 'error'
            }

# Initialize the agent
agent = DataAnalystAgent()

@app.route('/api/', methods=['POST'])
def analyze_data():
    """Main API endpoint for data analysis"""
    start_time = datetime.now()
    
    try:
        # Parse questions from questions.txt
        questions = []
        questions_file = request.files.get('questions.txt')
        
        if questions_file:
            questions_content = questions_file.read().decode('utf-8')
            questions = [q.strip() for q in questions_content.split('\n') if q.strip()]
        
        if not questions:
            return jsonify({
                'error': 'No questions found in questions.txt',
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Load all data files
        available_data = {}
        
        for file_key in request.files:
            if file_key != 'questions.txt':
                file = request.files[file_key]
                if file and file.filename:
                    # Save file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                        file.save(temp_file.name)
                        
                        # Load data
                        df = agent.load_data_from_file(temp_file.name, file.filename)
                        if df is not None:
                            available_data[file.filename] = df
                        
                        # Clean up temp file
                        os.unlink(temp_file.name)
        
        # Process each question
        results = []
        for i, question in enumerate(questions):
            # Check timeout (leave 30 seconds buffer)
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > 270:  # 4.5 minutes
                logger.warning(f"Timeout reached, stopping at question {i+1}")
                break
                
            answer = agent.answer_question(question, available_data)
            results.append(answer)
        
        response = {
            'results': results,
            'total_questions': len(questions),
            'processed_questions': len(results),
            'data_files_loaded': list(available_data.keys()),
            'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'processing_time_seconds': (datetime.now() - start_time).total_seconds()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large',
        'status': 'error',
        'max_size_mb': 50
    }), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)

