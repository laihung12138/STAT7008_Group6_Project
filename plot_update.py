# Package installation check
import subprocess
import sys
import pkg_resources

# List of required packages
required_packages = [
   'streamlit',
   'pandas',
   'numpy',
   'scikit-learn',
   'plotly',
   'seaborn',
   'scipy',
   'watchdog'
]

# Check and install packages
print("Checking required packages...")
missing_packages = []

# Find missing packages
for package in required_packages:
   try:
       pkg_resources.get_distribution(package)
       print(f"{package} is already installed")
   except pkg_resources.DistributionNotFound:
       missing_packages.append(package)

# Install missing packages if any
if missing_packages:
   print("\nMissing packages:", missing_packages)
   print("\nInstalling missing packages...")
   for package in missing_packages:
       try:
           subprocess.check_call([sys.executable, "-m", "pip", "install", package])
           print(f"Successfully installed {package}")
       except Exception as e:
           print(f"Error installing {package}: {e}")
else:
   print("\nAll required packages are already installed!")

# Standard library imports
import os
import json
import time
import threading
from typing import Dict, Any, Tuple
from datetime import datetime

# Data processing and analysis
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import seaborn as sns

# File monitoring
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import tempfile
import plot_update
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from plotly.graph_objects import Sankey
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates

class FileUpdateHandler(FileSystemEventHandler):
   def __init__(self, app):
       self.app = app
       
   def on_modified(self, event):
       if not event.is_directory and event.src_path.endswith(('.csv', '.xlsx', '.xls', '.json')):
           self.app.file_update_flag = True
           self.app.modified_file = event.src_path

class DataApp:
   def __init__(self):
       # Initialize session state for monitoring status if it doesn't exist
       if 'auto_update_enabled' not in st.session_state:
           st.session_state.auto_update_enabled = False
       if 'time_unit' not in st.session_state:
           st.session_state.time_unit = 'minutes'
       self.data = None
       self.preprocessed_data = None
       self.numeric_cols = []
       self.categorical_cols = []
       self.datetime_cols = []
       self.feature_names = []
       self.target_column = None
       self.outlier_stats = {}
       self.preprocessed_stats = {}
       self.last_update = None
       self.update_interval = 30
       self.file_path = None
       self.file_type = None
       self.file_update_flag = False
       self.modified_file = None
       self.observer = None
       self.update_thread = None
       self.is_updating = False
       self.watch_directory = None
       self.supported_files = {
           '.csv': 'CSV file',
           '.xlsx': 'Excel file (XLSX)',
           '.xls': 'Excel file (XLS)',
           '.json': 'JSON file'
       }

       
   def convert_to_seconds(self, value, unit):
       """Convert time value to seconds based on unit"""
       if unit == 'minutes':
           return value * 60
       elif unit == 'hours':
           return value * 3600
       return value
   
   def setup_file_monitoring(self, file_path):
       """Set up file monitoring for the given path"""
       try:
           self.file_path = file_path
           self.watch_directory = os.path.dirname(file_path)
           self.file_type = os.path.splitext(file_path)[1].lower()
           
           # Initial data load
           self.load_data()
           
           # Set up file observer
           if self.observer is None:
               self.observer = Observer()
               handler = FileUpdateHandler(self)
               self.observer.schedule(handler, self.watch_directory, recursive=False)
               self.observer.start()
           
           # Start update thread if not already running
           if self.update_thread is None or not self.update_thread.is_alive():
               self.is_updating = True
               self.update_thread = threading.Thread(target=self._update_loop)
               self.update_thread.daemon = True
               self.update_thread.start()
               
           st.success(f"Started monitoring: {file_path}")
           
       except Exception as e:
           st.error(f"Error setting up file monitoring: {str(e)}")

   def start_monitoring(self):
       """Start the monitoring process"""
       self.setup_file_monitoring(self.file_path)
       st.session_state.auto_update_enabled = True

   def stop_monitoring(self):
       """Stop the monitoring process"""
       self.is_updating = False
       if self.observer:
           self.observer.stop()
           self.observer.join()
       if self.update_thread:
           self.update_thread.join()
       self.observer = None
       self.update_thread = None
       st.session_state.auto_update_enabled = False

           
   def load_data(self):
       """Load data from the specified file"""
       try:
           if not os.path.exists(self.file_path):
               raise FileNotFoundError(f"File not found: {self.file_path}")
           
           # Create a lock to prevent concurrent file access
           with threading.Lock():
               if self.file_type == '.csv':
                   df = pd.read_csv(self.file_path)
               elif self.file_type in ['.xlsx', '.xls']:
                   df = pd.read_excel(self.file_path)
               elif self.file_type == '.json':
                   df = pd.read_json(self.file_path)
               else:
                   raise ValueError(f"Unsupported file type: {self.file_type}")
           
           # Only attempt datetime conversion for columns with date-related names
           date_indicators = ['date', 'time', 'year', 'month', 'day']
           for col in df.select_dtypes(include=['object']).columns:
               if any(indicator in col.lower() for indicator in date_indicators):
                   try:
                       # Test conversion on a sample first
                       sample = df[col].head(100)
                       test_conversion = pd.to_datetime(sample, errors='coerce')
                       if test_conversion.isnull().sum() / len(test_conversion) < 0.2:
                           df[col] = pd.to_datetime(df[col], errors='coerce')
                   except:
                       continue
           
           # Update column categories
           self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
           self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
           self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
           
           self.data = df
           self.last_update = datetime.now()
           
           return self.data
           
       except Exception as e:
           st.error(f"Error loading file: {str(e)}")
           return None


   def render_monitoring_controls(self):
       """Render controls for real-time monitoring"""
       st.sidebar.subheader("Real-time Monitoring")
       
       col1, col2 = st.sidebar.columns(2)
       
       with col1:
           time_unit = st.selectbox(
               "Time Unit",
               options=['minutes', 'hours'],
               key='time_unit'
           )

       with col2:
           if time_unit == 'minutes':
               update_value = st.number_input(
                   "Update Interval",
                   min_value=0.5,
                   max_value=60.0,
                   value=1.0,
                   step=0.5,
                   help="How often to check for file updates (in minutes)"
               )
           else:  # hours
               update_value = st.number_input(
                   "Update Interval",
                   min_value=0.5,
                   max_value=24.0,
                   value=1.0,
                   step=0.5,
                   help="How often to check for file updates (in hours)"
               )
           
           # Convert to seconds
           new_interval = self.convert_to_seconds(update_value, time_unit)
           if new_interval != self.update_interval:
               self.update_interval = new_interval

       # Display the actual interval in human-readable format
       if self.update_interval < 60:
           interval_text = f"{self.update_interval} seconds"
       elif self.update_interval < 3600:
           minutes = self.update_interval / 60
           interval_text = f"{minutes:.1f} minutes"
       else:
           hours = self.update_interval / 3600
           interval_text = f"{hours:.1f} hours"
       
       st.sidebar.caption(f"Current update interval: {interval_text}")
       
       # Auto-update controls
       col3, col4 = st.sidebar.columns(2)
       with col3:
           if not st.session_state.auto_update_enabled:
               if st.button("▶️ Start Auto Update", use_container_width=True):
                   self.start_monitoring()
                   st.rerun()
       with col4:
           if st.session_state.auto_update_enabled:
               if st.button("⏹️ Stop Auto Update", use_container_width=True):
                   self.stop_monitoring()
                   st.rerun()
       
       # Show monitoring status using session state
       status_color = "🟢" if st.session_state.auto_update_enabled else "🔴"
       st.sidebar.markdown(f"{status_color} Monitoring Status: "
                         f"{'Active' if st.session_state.auto_update_enabled else 'Inactive'}")
       
       if st.session_state.auto_update_enabled and self.last_update:
           st.sidebar.markdown(f"Last update: {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
           
   def _update_loop(self):
       """Background loop to check for file updates"""
       while self.is_updating:
           try:
               if self.file_update_flag:
                   if self.load_data():
                       st.rerun() 
                   self.file_update_flag = False
                   
               time.sleep(self.update_interval)  # Use user-defined interval
               
           except Exception as e:
               st.error(f"Error in update loop: {str(e)}")
               time.sleep(5)  # Wait longer on error

   def check_for_updates(self):
       """Check if the file has been modified"""
       try:
           if not self.file_path or not os.path.exists(self.file_path):
               return False
               
           current_mtime = os.path.getmtime(self.file_path)
           last_mtime = os.path.getmtime(self.modified_file) if self.modified_file else 0
           
           return current_mtime > last_mtime
           
       except Exception as e:
           st.error(f"Error checking for updates: {str(e)}")
           return False

   def validate_data(self):
       """Validate uploaded data"""
       if self.data is None:
           return False
       if self.data.empty:
           st.error("The uploaded file contains no data")
           return False
       if len(self.data.columns) < 2:
           st.error("The dataset must have at least two columns")
           return False
       return True

   def _preprocess_data(self):
       """Preprocess the data"""
       df = self.data.copy()
       
       # Handle missing values
       numeric_cols = df.select_dtypes(include=[np.number]).columns
       categorical_cols = df.select_dtypes(include=[object]).columns
       
       # Numeric columns
       for col in numeric_cols:
           # Fill missing with interpolate
           df[col].interpolate(method="linear", inplace=True)
           
           # Handle outliers using IQR
           Q1 = df[col].quantile(0.25)
           Q3 = df[col].quantile(0.75)
           IQR = Q3 - Q1
           lower_bound = Q1 - 1.5 * IQR
           upper_bound = Q3 + 1.5 * IQR
           df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
           df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
       
       # Categorical columns
       for col in categorical_cols:
           df[col].fillna(df[col].mode()[0], inplace=True)
           df[col] = df[col].str.lower().str.strip()
       
       return df

   

   def preprocess_features(self, scaling_method='standard', encoding_method='label'):
       """Preprocess features with multiple scaling and encoding options"""
       try:
           if self.preprocessed_data is None:
               self._preprocess_data()
           
           df = self.preprocessed_data.copy()
           
           # Numeric feature scaling
           if len(self.numeric_cols) > 0:
               if scaling_method == 'standard':
                   scaler = StandardScaler()
               elif scaling_method == 'minmax':
                   scaler = MinMaxScaler()
               elif scaling_method == 'robust':
                   scaler = RobustScaler()
               else:
                   raise ValueError(f"Unknown scaling method: {scaling_method}")
               
               df[self.numeric_cols] = scaler.fit_transform(df[self.numeric_cols])
               self.scaler = scaler
           
           # Categorical feature encoding
           if len(self.categorical_cols) > 0:
               if encoding_method == 'label':
                   for col in self.categorical_cols:
                       df[col] = pd.Categorical(df[col]).codes
                       
               elif encoding_method == 'onehot':
                   df = pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)
                   
               elif encoding_method == 'target' and self.target_column:
                   for col in self.categorical_cols:
                       global_mean = df[self.target_column].mean()
                       agg = df.groupby(col)[self.target_column].agg(['mean', 'count'])
                       smoothing = 1 / (1 + np.exp(-(agg['count'] - 10) / 10))
                       df[f'{col}_encoded'] = df[col].map(agg['mean'] * smoothing + global_mean * (1 - smoothing))
                       df.drop(col, axis=1, inplace=True)
           
           self.feature_names = df.columns.tolist()
           return df
           
       except Exception as e:
           st.error(f"Error in feature preprocessing: {str(e)}")
           return self.preprocessed_data.copy()
   
   def _set_cols(self):
       """Reset and set column types"""
       self.numeric_cols = []
       self.categorical_cols = []
       self.datetime_cols = []

       for col in self.data.columns:
           if pd.api.types.is_numeric_dtype(self.data[col]):
               self.numeric_cols.append(col)
           elif pd.api.types.is_datetime64_dtype(self.data[col]):
               self.datetime_cols.append(col)
           else:
               try:
                   pd.to_datetime(self.data[col])
                   self.datetime_cols.append(col)
               except:
                   self.categorical_cols.append(col)


   def info(self) -> Dict[str, Any]:
       """Get dataset info"""
       return {
           "Rows": len(self.data),
           "Columns": len(self.data.columns),
           "Numeric": len(self.numeric_cols),  # Changed from num_cols
           "Categorical": len(self.categorical_cols),  # Changed from cat_cols
           "Dates": len(self.datetime_cols),  # Changed from date_cols
           "Missing": self.data.isnull().sum().sum(),
           "Memory (MB)": round(self.data.memory_usage().sum() / 1024 / 1024, 2)
       }
   
   def analyze_mixed_data(self) -> Tuple[pd.DataFrame, Dict, Dict]:
       """Analyze mixed categorical and numerical data"""
       numerical_summary = self.data[self.numeric_cols].describe()
       categorical_summary = {col: self.data[col].value_counts() 
                            for col in self.categorical_cols}

       cross_tabs = {}
       for cat_col in self.categorical_cols:
           for num_col in self.numeric_cols:
               cross_tabs[f"{cat_col}_{num_col}"] = \
                   self.data.groupby(cat_col)[num_col].agg(['mean', 'count'])

       return numerical_summary, categorical_summary, cross_tabs

   
   def plot_num(self, column):
       st.subheader(f"Numerical Analysis: {column}")

       tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Distribution", "Box Plot", "Violin Plot", "Statistics", "Q-Q Plot", "Correlation Heatmap"])

       with tab1:
           fig = go.Figure()
           fig.add_trace(go.Histogram(x=self.data[column], name='Histogram'))
           fig.add_trace(go.Scatter(x=self.data[column].sort_values(),
                                  y=stats.norm.pdf(sorted(self.data[column])),
                                  name='Normal Distribution'))
           fig.update_layout(title=f'Distribution of {column}',
                           xaxis_title=column,
                           yaxis_title='Frequency')
           st.plotly_chart(fig)

       with tab2:
           fig = go.Figure()
           fig.add_trace(go.Box(y=self.data[column], name=column))
           fig.update_layout(title=f'Box Plot of {column}')
           st.plotly_chart(fig)

       with tab3:
           fig = go.Figure()
           fig.add_trace(go.Violin(y=self.data[column], name=column))
           fig.update_layout(title=f'Violin Plot of {column}')
           st.plotly_chart(fig)

       with tab4:
           col1, col2 = st.columns(2)
           with col1:
               stats_df = pd.DataFrame({
                   'Statistic': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis'],
                   'Value': [
                       self.data[column].mean(),
                       self.data[column].median(),
                       self.data[column].std(),
                       self.data[column].skew(),
                       self.data[column].kurtosis()
                   ]
               })
               st.dataframe(stats_df)

           with col2:
               quantiles_df = pd.DataFrame({
                   'Quantile': ['25%', '50%', '75%', 'IQR'],
                   'Value': [
                       self.data[column].quantile(0.25),
                       self.data[column].quantile(0.50),
                       self.data[column].quantile(0.75),
                       self.data[column].quantile(0.75) - self.data[column].quantile(0.25)
                   ]
               })
               st.dataframe(quantiles_df)

       with tab5:
           # Q-Q Plot
           fig = go.Figure()
           data = self.data[column].dropna()
           theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
           sorted_data = np.sort(data)

           fig.add_trace(go.Scatter(
               x=theoretical_quantiles,
               y=sorted_data,
               mode='markers',
               name='Data Points'
           ))

           # Add reference line
           min_val = min(theoretical_quantiles)
           max_val = max(theoretical_quantiles)
           fig.add_trace(go.Scatter(
               x=[min_val, max_val],
               y=[min_val * data.std() + data.mean(), max_val * data.std() + data.mean()],
               mode='lines',
               name='Reference Line',
               line=dict(color='red', dash='dash')
           ))

           fig.update_layout(
               title='Q-Q Plot (Normal Distribution)',
               xaxis_title='Theoretical Quantiles',
               yaxis_title='Sample Quantiles'
           )
           st.plotly_chart(fig)
   
       with tab6:
           if len(self.numeric_cols) > 1:
               # Correlation matrix for numeric columns
               corr_matrix = self.data[self.numeric_cols].corr()
           
               # Create heatmap
               fig = go.Figure(data=go.Heatmap(
                   z=corr_matrix.values,
                   x=corr_matrix.columns,
                   y=corr_matrix.columns,
                   colorscale='RdBu',
                   zmin=-1,
                   zmax=1,
                   text=np.round(corr_matrix.values, 2),
                   texttemplate='%{text}',
                   textfont={"size": 10},
                   hoverongaps=False
               ))

               fig.update_layout(
                   title=f'Correlation Heatmap (Highlighting {column})',
                   width=800,
                   height=800
               )
               st.plotly_chart(fig)

               # Display strongest correlations with selected column
               st.subheader(f"Strongest Correlations with {column}")
               correlations = corr_matrix[column].sort_values(ascending=False)
               correlations = correlations[correlations.index != column]  # Remove self-correlation
               corr_df = pd.DataFrame({
                   'Variable': correlations.index,
                   'Correlation': correlations.values
               })
               st.dataframe(corr_df)
           else:
               st.warning("Need at least 2 numeric columns for correlation analysis")
           
   def plot_cat(self, column):
       st.subheader(f"Categorical Analysis: {column}")
       
       tab1, tab2, tab3 = st.tabs(["Bar Chart", "Pie Chart", "Statistics"])
       
       value_counts = self.data[column].value_counts()
       
       with tab1:
           fig = px.bar(x=value_counts.index, y=value_counts.values,
                       title=f'Distribution of {column}')
           st.plotly_chart(fig)

       with tab2:
           fig = px.pie(values=value_counts.values, names=value_counts.index,
                       title=f'Distribution of {column}')
           st.plotly_chart(fig)

       with tab3:
           stats_df = pd.DataFrame({
               'Statistic': ['Unique Values', 'Most Common', 'Least Common', 'Missing Values'],
               'Value': [
                   len(value_counts),
                   f"{value_counts.index[0]} ({value_counts.values[0]})",
                   f"{value_counts.index[-1]} ({value_counts.values[-1]})",
                   self.data[column].isnull().sum()
               ]
           })
           st.dataframe(stats_df)

   def analyze_time_series(self, date_col: str, value_col: str) -> Tuple[pd.DataFrame, Dict]:
       """
       Analyze time series data including trends and seasonality with automatic date detection

       """
       try:
           df = self.data.copy()

           # Verify and convert date column
           if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
               df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
               if df[date_col].isnull().sum() > 0:
                   raise ValueError(f"Could not convert all values in {date_col} to dates")

           df = df.sort_values(date_col)
           analysis_results = {}

           # Calculate moving averages with dynamic window sizes
           windows = [7, 30, 90]
           for window in windows:
               if len(df) >= window:
                   df[f'MA{window}'] = df[value_col].rolling(window=window, min_periods=1).mean()

           # Seasonality analysis
           df['year'] = df[date_col].dt.year
           df['month'] = df[date_col].dt.month

           # Enhanced analysis results
           analysis_results.update({
               'monthly_avg': df.groupby('month')[value_col].mean(),
               'yearly_avg': df.groupby('year')[value_col].mean(),
               'basic_stats': df[value_col].describe(),
               'trend': df[value_col].diff().mean(),
               'volatility': df[value_col].std()
           })

           return df, analysis_results
       except Exception as e:
           raise Exception(f"Error in time series analysis: {str(e)}")

   def plot_time_series(self, date_column: str, value_column: str):
       """
       Create interactive time series visualizations with automatic date detection
       """
       try:
           # Verify columns exist
           if date_column not in self.data.columns or value_column not in self.data.columns:
               st.error("Selected columns not found in dataset")
               return

           df, analysis_results = self.analyze_time_series(date_column, value_column)

           # Create tabs for different visualizations
           tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Moving Averages", "Seasonality", "Statistics"])

           with tab1:
               fig = px.line(df, x=date_column, y=value_column,
                            title=f'{value_column} over Time')
               fig.update_layout(template='plotly_white')
               st.plotly_chart(fig, use_container_width=True)

               # Add basic trend information
               trend = analysis_results['trend']
               st.info(f"Overall trend: {'Increasing' if trend > 0 else 'Decreasing' if trend < 0 else 'Stable'} "
                      f"(Average change: {trend:.2f})")

           with tab2:
               fig = go.Figure()
               fig.add_trace(go.Scatter(
                   x=df[date_column], 
                   y=df[value_column],
                   name='Original',
                   line=dict(color='gray', width=1)
               ))

               colors = ['red', 'green', 'blue']
               windows = [7, 30, 90]
               for window, color in zip(windows, colors):
                   if f'MA{window}' in df.columns:
                       fig.add_trace(go.Scatter(
                           x=df[date_column],
                           y=df[f'MA{window}'],
                           name=f'{window}-day MA',
                           line=dict(color=color)
                       ))

               fig.update_layout(
                   title=f'Moving Averages of {value_column}',
                   xaxis_title='Date',
                   yaxis_title=value_column,
                   template='plotly_white'
               )
               st.plotly_chart(fig, use_container_width=True)

           with tab3:
               if len(df) >= 2:
                   col1, col2 = st.columns(2)

                   with col1:
                       # Monthly seasonality
                       monthly_avg = analysis_results['monthly_avg']
                       fig = px.line(
                           x=monthly_avg.index, 
                           y=monthly_avg.values,
                           title=f'Monthly Seasonality of {value_column}',
                           markers=True
                       )
                       fig.update_layout(
                           xaxis_title='Month',
                           yaxis_title=f'Average {value_column}',
                           template='plotly_white'
                       )
                       st.plotly_chart(fig, use_container_width=True)

                   with col2:
                       # Yearly trend
                       yearly_avg = analysis_results['yearly_avg']
                       fig = px.line(
                           x=yearly_avg.index, 
                           y=yearly_avg.values,
                           title=f'Yearly Trend of {value_column}',
                           markers=True
                       )
                       fig.update_layout(
                           xaxis_title='Year',
                           yaxis_title=f'Average {value_column}',
                           template='plotly_white'
                       )
                       st.plotly_chart(fig, use_container_width=True)

           with tab4:
               # Display comprehensive statistics
               st.subheader("Statistical Summary")
               col1, col2 = st.columns(2)

               with col1:
                   st.write("Basic Statistics:")
                   st.dataframe(analysis_results['basic_stats'])

               with col2:
                   st.write("Additional Metrics:")
                   metrics = {
                       "Volatility": analysis_results['volatility'],
                       "Total Change": df[value_column].iloc[-1] - df[value_column].iloc[0],
                       "Average Value": df[value_column].mean(),
                       "Data Range": f"{df[date_column].min().date()} to {df[date_column].max().date()}"
                   }
                   for key, value in metrics.items():
                       st.metric(key, f"{value:.2f}" if isinstance(value, float) else value)

       except Exception as e:
           st.error(f"Error in time series visualization: {str(e)}")
           st.exception(e)


   def render_time_series_section(self):
       st.subheader("Time Series Analysis")

       # Detect potential date columns with stricter validation
       potential_date_cols = []
       for col in self.data.columns:
           try:
               # Check if column is already datetime
               if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                   potential_date_cols.append(col)
               else:
                   # Try converting a sample to datetime
                   sample = self.data[col].head(100)  # Check first 100 rows
                   test_conversion = pd.to_datetime(sample, errors='coerce')

                   # Additional validation checks
                   if (
                       # Check if conversion was successful
                       test_conversion.isnull().sum() / len(test_conversion) < 0.2 
                       # Check if column name suggests it's a date
                       and any(date_indicator in col.lower() 
                              for date_indicator in ['date', 'time', 'year', 'month', 'day'])
                   ):
                       potential_date_cols.append(col)
           except:
               continue

       if not potential_date_cols:
           st.warning("No date columns detected. Please ensure your data contains columns with date information (column names should include 'date', 'time', 'year', 'month', or 'day').")
           return

       # Date column selection
       date_col = st.selectbox(
           "Select Date Column",
           options=potential_date_cols,
           help="Select the column containing date information"
       )

       # Preview date conversion before proceeding
       try:
           sample_dates = pd.to_datetime(self.data[date_col].head(), errors='coerce')
           st.write("Preview of date conversion:", sample_dates)

           if sample_dates.isnull().all():
               st.error("Unable to convert selected column to dates. Please select a different column.")
               return

       except Exception as e:
           st.error(f"Error converting dates: {str(e)}")
           return

       # Get numeric columns for value selection
       numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns

       if len(numeric_cols) == 0:
           st.warning("No numeric columns available for analysis.")
           return

       # Value column selection
       value_col = st.selectbox(
           "Select Value Column",
           options=numeric_cols,
           help="Select the numeric column to analyze over time"
       )

       try:
           # Convert date column to datetime if it's not already
           if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
               conversion_test = pd.to_datetime(self.data[date_col], errors='coerce')
               if conversion_test.isnull().sum() / len(conversion_test) > 0.2:
                   st.error(f"Unable to convert {date_col} to valid dates. Too many invalid values.")
                   return

               self.data[date_col] = conversion_test
               st.success(f"Successfully converted {date_col} to datetime format")

           # Add analysis options
           st.subheader("Analysis Options")
           col1, col2 = st.columns(2)

           with col1:
               resample_period = st.selectbox(
                   "Resample Period (Optional)",
                   ["None", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
                   help="Resample data to reduce noise and see trends more clearly"
               )

           with col2:
               handle_missing = st.selectbox(
                   "Handle Missing Values",
                   ["Drop", "Forward Fill", "Backward Fill", "Linear Interpolation"],
                   help="Choose how to handle missing values in the time series"
               )

           if st.button("Generate Time Series Analysis"):
               # Create a copy of the data for analysis
               analysis_df = self.data[[date_col, value_col]].copy()
               analysis_df = analysis_df.sort_values(date_col)

               # Handle missing values
               if handle_missing == "Drop":
                   analysis_df = analysis_df.dropna()
               elif handle_missing == "Forward Fill":
                   analysis_df = analysis_df.fillna(method='ffill')
               elif handle_missing == "Backward Fill":
                   analysis_df = analysis_df.fillna(method='bfill')
               elif handle_missing == "Linear Interpolation":
                   analysis_df[value_col] = analysis_df[value_col].interpolate(method='linear')

               # Resample data if selected
               if resample_period != "None":
                   period_map = {
                       "Daily": "D",
                       "Weekly": "W",
                       "Monthly": "M",
                       "Quarterly": "Q",
                       "Yearly": "Y"
                   }
                   analysis_df = analysis_df.set_index(date_col)
                   analysis_df = analysis_df.resample(period_map[resample_period]).mean()
                   analysis_df.reset_index(inplace=True)
                   st.info(f"Data resampled to {resample_period.lower()} frequency")

               # Store the processed data temporarily for plotting
               temp_data = self.data.copy()
               self.data = analysis_df

               # Generate time series plots
               self.plot_time_series(date_col, value_col)

               # Restore original data
               self.data = temp_data

       except Exception as e:
           st.error(f"Error in time series analysis: {str(e)}")
           st.exception(e)







   def custom_visualization(self):
       if self.data is None:
           st.error("No data loaded")
           return

       st.subheader("Custom Visualization")

       # Plot type selection
       plot_type = st.selectbox(
           "Select Plot Type",
           ["Scatter Plot", "Line Plot", "Bar Plot", "Box Plot", "Violin Plot", 
            "Histogram", "Density Plot", "Heatmap", "Bubble Plot", "Area Plot"]
       )

       # Common parameters
       col1, col2 = st.columns(2)
       with col1:
           x_col = st.selectbox("Select X-axis", self.data.columns)
           color_col = st.selectbox("Select Color Variable (optional)", 
                                  ['None'] + list(self.data.columns))
       with col2:
           y_col = st.selectbox("Select Y-axis (optional)", 
                               ['None'] + list(self.data.columns))
           size_col = st.selectbox("Select Size Variable (optional)", 
                                 ['None'] + list(self.data.columns))

       # Plot specific parameters
       with st.expander("Plot Settings"):
           title = st.text_input("Plot Title", "Custom Plot")

           col1, col2 = st.columns(2)
           with col1:
               x_title = st.text_input("X-axis Title", x_col)
               marker_size = st.slider("Marker Size", 2, 20, 6)
               opacity = st.slider("Opacity", 0.0, 1.0, 0.7)
           with col2:
               y_title = st.text_input("Y-axis Title", y_col if y_col != 'None' else '')
               plot_height = st.slider("Plot Height", 400, 1000, 600)
               plot_width = st.slider("Plot Width", 400, 1200, 800)

           color_scheme = st.selectbox(
               "Color Scheme",
               ["Viridis", "Plasma", "Inferno", "Magma", "RdBu", "Spectral"]
           )

           show_trend = st.checkbox("Show Trend Line", False)

       # Create figure based on plot type
       fig = go.Figure()

       try:
           if plot_type == "Scatter Plot":
               if y_col == 'None':
                   st.error("Please select Y-axis for scatter plot")
                   return

               scatter_data = dict(
                   x=self.data[x_col],
                   y=self.data[y_col],
                   mode='markers',
                   name='Data Points'
               )

               if color_col != 'None':
                   scatter_data['color'] = self.data[color_col]
               if size_col != 'None':
                   scatter_data['size'] = self.data[size_col]

               fig = px.scatter(**scatter_data, 
                              title=title,
                              opacity=opacity,
                              color_continuous_scale=color_scheme)

               if show_trend:
                   fig.add_traces(px.scatter(x=self.data[x_col], 
                                           y=self.data[y_col], 
                                           trendline="ols").data)

           elif plot_type == "Line Plot":
               if y_col == 'None':
                   st.error("Please select Y-axis for line plot")
                   return

               fig = px.line(self.data, x=x_col, y=y_col, 
                            color=None if color_col == 'None' else color_col,
                            title=title)

           elif plot_type == "Bar Plot":
               if y_col == 'None':
                   # Single variable bar plot
                   counts = self.data[x_col].value_counts()
                   fig = px.bar(x=counts.index, y=counts.values)
               else:
                   fig = px.bar(self.data, x=x_col, y=y_col,
                              color=None if color_col == 'None' else color_col)

           elif plot_type == "Box Plot":
               if y_col == 'None':
                   fig = px.box(self.data, x=x_col)
               else:
                   fig = px.box(self.data, x=x_col, y=y_col,
                              color=None if color_col == 'None' else color_col)

           elif plot_type == "Violin Plot":
               if y_col == 'None':
                   fig = px.violin(self.data, x=x_col)
               else:
                   fig = px.violin(self.data, x=x_col, y=y_col,
                                 color=None if color_col == 'None' else color_col)

           elif plot_type == "Histogram":
               fig = px.histogram(self.data, x=x_col,
                                color=None if color_col == 'None' else color_col,
                                opacity=opacity)

           elif plot_type == "Density Plot":
               if self.data[x_col].dtype.kind not in 'iufc':
                   st.error("Density plot requires numerical data")
                   return

               kde = stats.gaussian_kde(self.data[x_col].dropna())
               x_range = np.linspace(self.data[x_col].min(), self.data[x_col].max(), 100)
               fig = go.Figure(go.Scatter(x=x_range, y=kde(x_range), fill='tozeroy'))

           elif plot_type == "Heatmap":
               if y_col == 'None':
                   st.error("Please select Y-axis for heatmap")
                   return

               pivot_data = self.data.pivot_table(
                   values=size_col if size_col != 'None' else y_col,
                   index=y_col,
                   columns=x_col,
                   aggfunc='mean'
               )

               fig = px.imshow(pivot_data,
                             color_continuous_scale=color_scheme)

           elif plot_type == "Bubble Plot":
               if y_col == 'None' or size_col == 'None':
                   st.error("Please select Y-axis and Size variable for bubble plot")
                   return

               fig = px.scatter(self.data,
                              x=x_col,
                              y=y_col,
                              size=size_col,
                              color=None if color_col == 'None' else color_col,
                              color_continuous_scale=color_scheme)

           elif plot_type == "Area Plot":
               if y_col == 'None':
                   st.error("Please select Y-axis for area plot")
                   return

               fig = px.area(self.data,
                            x=x_col,
                            y=y_col,
                            color=None if color_col == 'None' else color_col)

           # Update layout for all plots
           fig.update_layout(
               title=title,
               xaxis_title=x_title,
               yaxis_title=y_title,
               height=plot_height,
               width=plot_width,
               showlegend=True
           )

           fig.update_traces(marker=dict(size=marker_size))

           # Display plot
           st.plotly_chart(fig)

           # Add plot description
           with st.expander("Plot Description"):
               st.write(f"**Plot Type:** {plot_type}")
               st.write(f"**X-axis:** {x_col}")
               st.write(f"**Y-axis:** {y_col if y_col != 'None' else 'Not specified'}")
               if color_col != 'None':
                   st.write(f"**Color Variable:** {color_col}")
               if size_col != 'None':
                   st.write(f"**Size Variable:** {size_col}")

               # Add correlations for scatter plots
               if plot_type == "Scatter Plot" and y_col != 'None':
                   if self.data[x_col].dtype.kind in 'iufc' and self.data[y_col].dtype.kind in 'iufc':
                       correlation = self.data[x_col].corr(self.data[y_col])
                       st.write(f"**Correlation Coefficient:** {correlation:.3f}")

       except Exception as e:
           st.error(f"Error creating plot: {str(e)}")
           st.write("Please check if your selected variables are compatible with the chosen plot type.")


   def safe_operation(self, operation, error_message):
       """Wrapper for safe operation execution"""
       try:
           return operation()
       except Exception as e:
           st.error(f"{error_message}: {str(e)}")
           return None
   

   def run(self):
       st.title("Data Analysis Dashboard")

       # File uploader with additional file types
       uploaded_file = st.sidebar.file_uploader("Upload File", type=['csv', 'xlsx', 'xls', 'json'])

       if uploaded_file is not None:
           try:
               # Load data based on file type
               if uploaded_file.name.endswith('.csv'):
                   self.data = pd.read_csv(uploaded_file)
               elif uploaded_file.name.endswith('.json'):
                   self.data = pd.read_json(uploaded_file)
               else:  # xlsx or xls
                   self.data = pd.read_excel(uploaded_file)

               # Display basic information
               st.sidebar.success("File successfully loaded!")

               # Save uploaded file temporarily
               # 这个地方我需要加
               file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
               with open(file_path, 'wb') as f:
                   f.write(uploaded_file.getbuffer())

               # Set file path and type
               self.file_path = file_path
               self.file_type = os.path.splitext(file_path)[1].lower()
           
               # Load initial data
               if self.load_data() is not None:
                   # Render monitoring controls only if data is loaded successfully
                   self.render_monitoring_controls()

               st.write("Data Preview:")
               st.dataframe(self.data.head())
               
               # Identify column types
               self.numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
               self.categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
               self.datetime_cols = self.data.select_dtypes(include=['datetime64']).columns

               # Create info dictionary - 信息 - 不用表，做成一列
               info = {
                   "Total Rows": self.data.shape[0],
                   "Total Columns": self.data.shape[1],
                   "Numeric Columns": len(self.numeric_cols),
                   "Categorical Columns": len(self.categorical_cols),
                   "DateTime Columns": len(self.datetime_cols),
                   "Missing Values": self.data.isnull().sum().sum(),
                   "Memory Usage": f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                   "Duplicated Rows": self.data.duplicated().sum()
               }

               # Create two columns for displaying info
               col1, col2 = st.columns(2)
               info_items = list(info.items())
               mid_point = len(info_items) // 2

               # Display in first column
               with col1:
                   for key, value in info_items[:mid_point]:
                       st.metric(label=key, value=value)

               # Display in second column
               with col2:
                   for key, value in info_items[mid_point:]:
                       st.metric(label=key, value=value)
                
                # 这个可以直接加
               # Main analysis selection
               analysis_type = st.sidebar.selectbox(
                   "Select Analysis Type",
                   ["Dataset Overview", "Numerical Analysis", "Categorical Analysis", 
                    "Time Series Analysis", "Custom Visualization"]
               )

               if analysis_type == "Dataset Overview":
                   st.subheader("Dataset Overview")
                   st.write(self.data.head())

                   # Display column information
                   col1, col2, col3 = st.columns(3)
                   with col1:
                       st.write("Numeric Columns:")
                       st.write(self.numeric_cols)
                   with col2:
                       st.write("Categorical Columns:")
                       st.write(self.categorical_cols)
                   with col3:
                       st.write("DateTime Columns:")
                       st.write(self.datetime_cols)

                   # Missing values analysis
                   st.subheader("Missing Values Analysis")
                   missing_data = pd.DataFrame({
                       'Column': self.data.columns,
                       'Missing Values': self.data.isnull().sum(),
                       'Percentage': (self.data.isnull().sum() / len(self.data) * 100).round(2)
                   })
                   st.dataframe(missing_data)

                   # Data summary
                   st.subheader("Data Summary")
                   st.write(self.data.describe())

               elif analysis_type == "Numerical Analysis":
                   st.subheader("Numerical Analysis")
                   if len(self.numeric_cols) == 0:
                       st.warning("No numerical columns found in the dataset")
                   else:
                       selected_num_col = st.selectbox("Select Numerical Column", self.numeric_cols)
                       self.plot_num(selected_num_col)

               elif analysis_type == "Categorical Analysis":
                   st.subheader("Categorical Analysis")
                   if len(self.categorical_cols) == 0:
                       st.warning("No categorical columns found in the dataset")
                   else:
                       selected_cat_col = st.selectbox("Select Categorical Column", self.categorical_cols)
                       self.plot_cat(selected_cat_col)

               elif analysis_type == "Time Series Analysis":
                   st.subheader("Time Series Analysis")
                   self.render_time_series_section()
           
               elif analysis_type == "Custom Visualization":
                   st.subheader("Custom Visualization")
                   self.custom_visualization()

           except Exception as e:
               st.error(f"Error: {str(e)}")
               st.write("Please check your file format and try again.")

       else:
           st.info("Please upload a CSV, Excel, or JSON file to begin analysis.")


if __name__ == "__main__":
   app = DataApp()
   app.run()



# #### Plot Function about Bivariate and Multivariate 

# In[22]:




######################## Bivariate ########################

# Numerical vs Numerical Data
def plot_2d_density(data, x_col, y_col, key_suffix=""):
    fig = px.density_heatmap(data, x=x_col, y=y_col)
    fig.update_layout(title=f"2D Density Plot: {x_col} vs {y_col}", xaxis_title=x_col, yaxis_title=y_col)
    st.plotly_chart(fig, key=f"2d_density_{x_col}_{y_col}_{key_suffix}")

def plot_scatter(data, x_col, y_col, key_suffix=""):
    fig = px.scatter(data, x=x_col, y=y_col)
    st.plotly_chart(fig, key=f"scatter_{x_col}_{y_col}_{key_suffix}")

def plot_line(data, x_col, y_col, key_suffix=""):
    fig = px.line(data, x=x_col, y=y_col)
    st.plotly_chart(fig, key=f"line_{x_col}_{y_col}_{key_suffix}")

def plot_joint(data, x_col, y_col, key_suffix=""):
    fig = px.scatter(data, x=x_col, y=y_col)
    st.plotly_chart(fig, key=f"joint_{x_col}_{y_col}_{key_suffix}")

def plot_marginal(data, x_col, y_col, key_suffix=""):
    fig = px.scatter(data, x=x_col, y=y_col, marginal_x="histogram", marginal_y="histogram")
    st.plotly_chart(fig, key=f"marginal_{x_col}_{y_col}_{key_suffix}")

# Categorical vs Numerical Data
def box_plot(data, category_col, value_col, key_suffix=""):
    fig = px.box(data, x=category_col, y=value_col)
    st.plotly_chart(fig, key=f"box_plot_{category_col}_{value_col}_{key_suffix}")

def violin_plot(data, category_col, value_col, key_suffix=""):
    fig = px.violin(data, x=category_col, y=value_col, box=True)
    st.plotly_chart(fig, key=f"violin_plot_{category_col}_{value_col}_{key_suffix}")

def bar_plot(data, category_col, value_col, key_suffix=""):
    fig = px.bar(data, x=category_col, y=value_col)
    st.plotly_chart(fig, key=f"bar_plot_{category_col}_{value_col}_{key_suffix}")

def sunburst_chart(data, category_col, value_col, key_suffix=""):
    fig = px.sunburst(data, path=[category_col], values=value_col)
    st.plotly_chart(fig, key=f"sunburst_{category_col}_{value_col}_{key_suffix}")

# Categorical vs Categorical Data
def mosaic_plot(data, category1, category2, key_suffix=""):
    mosaic_data = pd.crosstab(data[category1], data[category2])
    fig, _ = mosaic(mosaic_data.stack(), title=f"Mosaic Plot: {category1} vs {category2}")
    st.pyplot(fig, key=f"mosaic_{category1}_{category2}_{key_suffix}")

def heatmap(data, category1, category2, key_suffix=""):
    pivot_data = data.pivot_table(index=category1, columns=category2)
    fig = px.imshow(pivot_data, text_auto=True)
    st.plotly_chart(fig, key=f"heatmap_{category1}_{category2}_{key_suffix}")

# Time Series Data
def line_plot(data, date_col, value_col, key_suffix=""):
    fig = px.line(data, x=date_col, y=value_col)
    st.plotly_chart(fig, key=f"line_plot_{date_col}_{value_col}_{key_suffix}")

def moving_average_plot(data, date_col, value_col, window_size=7, key_suffix=""):
    data['Moving Average'] = data[value_col].rolling(window=window_size).mean()
    fig = px.line(data, x=date_col, y=[value_col, 'Moving Average'])
    st.plotly_chart(fig, key=f"moving_avg_{date_col}_{value_col}_{window_size}_{key_suffix}")

def calendar_heatmap(data, date_col, value_col, key_suffix=""):
    data['Year'] = data[date_col].dt.year
    data['Month'] = data[date_col].dt.month
    pivot_data = data.pivot_table(index='Month', columns='Year', values=value_col, aggfunc='sum')
    fig = px.imshow(pivot_data)
    st.plotly_chart(fig, key=f"calendar_heatmap_{date_col}_{value_col}_{key_suffix}")

def time_series_scatter(data, date_col, value_col, key_suffix=""):
    fig = px.scatter(data, x=date_col, y=value_col)
    st.plotly_chart(fig, key=f"time_series_scatter_{date_col}_{value_col}_{key_suffix}")

def area_plot(data, date_col, value_col, key_suffix=""):
    fig = px.area(data, x=date_col, y=value_col)
    st.plotly_chart(fig, key=f"area_plot_{date_col}_{value_col}_{key_suffix}")

def waterfall_chart(data, date_col, value_col, key_suffix=""):
    cumulative_sum = data[value_col].cumsum()
    data['Cumulative'] = cumulative_sum
    fig = px.bar(data, x=date_col, y=value_col)
    st.plotly_chart(fig, key=f"waterfall_chart_{date_col}_{value_col}_{key_suffix}")

def dual_axis_line_plot(data, date_col, value_col1, value_col2, key_suffix=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[date_col], y=data[value_col1], name=value_col1))
    fig.add_trace(go.Scatter(x=data[date_col], y=data[value_col2], name=value_col2, yaxis="y2"))
    fig.update_layout(title="Dual-Axis Line Plot", yaxis2=dict(title=value_col2, overlaying='y', side='right'))
    st.plotly_chart(fig, key=f"dual_axis_line_{date_col}_{value_col1}_{value_col2}_{key_suffix}")

def plot_map(data, lat_lon_col, value_col):
    lat_col, lon_col = lat_lon_col
    print(lat_col, lon_col)
    fig = px.scatter_mapbox(data, lat=lat_col, lon=lon_col, color=value_col, size=value_col)
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig)

######################## Multivariate ########################

# Numerical Data
def plot_pair(data, columns, key_suffix=""):
    fig = px.scatter_matrix(data[columns])
    st.plotly_chart(fig, key=f"scatter_matrix_{columns}_{key_suffix}")

def plot_heatmap(data, columns, key_suffix=""):
    fig = px.imshow(data[columns].corr(), text_auto=True)
    fig.update_layout(title="Heatmap of Correlations")
    st.plotly_chart(fig, key=f"heatmap_{columns}_{key_suffix}")

def plot_bubble(data, x_col, y_col, size_col, key_suffix=""):
    fig = px.scatter(data, x=x_col, y=y_col, size=size_col)
    st.plotly_chart(fig, key=f"bubble_{x_col}_{y_col}_{size_col}_{key_suffix}")

# Categorical Data
def stacked_bar_plot(data, category_col, subcategory_col, value_col, key_suffix=""):
    fig = px.bar(data, x=category_col, y=value_col, color=subcategory_col, barmode="stack")
    st.plotly_chart(fig, key=f"stacked_bar_{category_col}_{subcategory_col}_{value_col}_{key_suffix}")

def grouped_bar_plot(data, category_col, subcategory_col, value_col, key_suffix=""):
    fig = px.bar(data, x=category_col, y=value_col, color=subcategory_col, barmode="group")
    st.plotly_chart(fig, key=f"grouped_bar_{category_col}_{subcategory_col}_{value_col}_{key_suffix}")

def grouped_violin_box(data, category_col, subcategory_col, value_col, plot_type="violin", key_suffix=""):
    if plot_type == "violin":
        fig = px.violin(data, x=category_col, y=value_col, color=subcategory_col, box=True)
    else:
        fig = px.box(data, x=category_col, y=value_col, color=subcategory_col)
    st.plotly_chart(fig, key=f"grouped_violin_box_{category_col}_{subcategory_col}_{value_col}_{plot_type}_{key_suffix}")

def clustered_bar_chart(data, category_col, subcategory_col, value_col, key_suffix=""):
    fig = px.bar(data, x=category_col, y=value_col, color=subcategory_col)
    st.plotly_chart(fig, key=f"clustered_bar_{category_col}_{subcategory_col}_{value_col}_{key_suffix}")

def bubble_chart(data, x_col, y_col, size_col, color_col, key_suffix=""):
    fig = px.scatter(data, x=x_col, y=y_col, size=size_col, color=color_col)
    st.plotly_chart(fig, key=f"bubble_chart_{x_col}_{y_col}_{size_col}_{color_col}_{key_suffix}")

def sankey_diagram(data, key_suffix=""):
    fig = go.Figure(go.Sankey(node=dict(pad=15, thickness=20), link=dict(source=[0, 1], target=[1, 2], value=[8, 4])))
    st.plotly_chart(fig, key=f"sankey_diagram_{key_suffix}")

# Time series data
def parallel_categories_plot(data, category_cols, key_suffix=""):
    # Ensure the key is unique for each chart rendering
    fig = px.parallel_categories(data, dimensions=category_cols)
    st.plotly_chart(fig, use_container_width=True, key=f"parallel_categories_{key_suffix}")

def candlestick_chart(data, date_col, open_col, high_col, low_col, close_col, key_suffix=""):
    fig = go.Figure(data=[go.Candlestick(
        x=data[date_col],
        open=data[open_col],
        high=data[high_col],
        low=data[low_col],
        close=data[close_col]
    )])
    st.plotly_chart(fig, use_container_width=True, key=f"candlestick_chart_{key_suffix}")


def time_series_correlation_matrix(data, value_cols, key_suffix=""):
    corr_matrix = data[value_cols].corr()
    fig = px.imshow(corr_matrix, title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True, key=f"time_series_corr_matrix_{key_suffix}")

