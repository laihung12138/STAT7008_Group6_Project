import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, df):
        """Initialize the data processor and load the dataset"""
        self.df = df

    def handle_exception_datatype(self):
        for column in self.df.select_dtypes(include=[np.number]).columns:
            # Replace non-numeric values with NaN
            self.df[column] = pd.to_numeric(self.df[column], errors='coerce')

    def handle_missing_values(self, method_number, method_object):
        """Handle missing values
        Parameters:
        method: str, 'mean', 'median', 'interpolate', 'mode', 'forward fill', 'backward fill' optional filling methods
        or delete directly
        """
        for column in self.df.select_dtypes(include=[np.number]).columns:
            if method_number == 'mean':
                self.df[column].fillna(self.df[column].mean(), inplace=True)
            elif method_number == 'median':
                self.df[column].fillna(self.df[column].median(), inplace=True)
            elif method_number == 'interpolate':
                self.df[column].interpolate(method="linear", inplace=True)
            elif method_number == 'mode':
                self.df[column].fillna(self.df[column].mode()[0], inplace=True)
            elif method_number == 'forward fill':
                self.df[column].ffill(inplace=True)
            elif method_number == 'backward fill':
                self.df[column].bfill(inplace=True)
            elif method_number == "delete":
                self.df.dropna(subset=[column], inplace=True)

        for column in self.df.select_dtypes(include=[object]).columns:
            if method_object == 'mode':
                self.df[column].fillna(self.df[column].mode()[0], inplace=True)
            elif method_object == 'forward fill':
                self.df[column].ffill(inplace=True)
            elif method_object == 'backward fill':
                self.df[column].bfill(inplace=True)
            elif method_object == "delete":
                self.df.dropna(subset=[column], inplace=True)

        return self.df

    def handle_outliers(self):
        """Handle outliers"""
        for column in self.df.select_dtypes(include=[np.number]).columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            self.df[column] = np.where(self.df[column] < lower_bound, lower_bound, self.df[column])
            self.df[column] = np.where(self.df[column] > upper_bound, upper_bound, self.df[column])

        return self.df

    def standardize_column_format(self, column_name, format_type='date'):
        """Standardize column format"""
        if format_type == 'date':
            self.df[column_name] = pd.to_datetime(self.df[column_name], errors='coerce')
        elif format_type == 'string':
            self.df[column_name] = self.df[column_name].str.lower().str.strip()

        return self.df

    def preprocess_data(self, date_column=None, category_column=None):
        """Main data preprocessing function"""
        self.handle_exception_datatype()
        self.handle_missing_values()
        self.handle_outliers()  
       
        if date_column and date_column in self.df.columns:
            self.standardize_column_format(date_column, format_type='date')

        if category_column and category_column in self.df.columns:
            self.standardize_column_format(category_column, format_type='string')
        
        return self.df
