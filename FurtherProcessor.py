import pandas as pd
from sklearn.preprocessing import StandardScaler

class FurtherProcessor:
    def __init__(self, df):
        """
        Initialize the FurtherProcessor class and import the CSV dataset.
        """
        self.data = df
 ######################################################################################################################
    def feature_engineering(self, columns, operation, new_name):
        """
        Perform addition, subtraction, multiplication, or division on specified columns.
        """
        if operation == 'add':
            self.data[f"{new_name}"] = self.data[columns[0]] + self.data[columns[1]]
        elif operation == 'subtract':
            self.data[f"{new_name}"] = self.data[columns[0]] - self.data[columns[1]]
        elif operation == 'multiply':
            self.data[f"{new_name}"] = self.data[columns[0]] * self.data[columns[1]]
        elif operation == 'divide':
            self.data[f"{new_name}"] = self.data[columns[0]] / self.data[columns[1]]
        else:
            raise ValueError("Unsupported operation. Please choose from 'add', 'subtract', 'multiply', or 'divide'.")
        return self.data
###########################################################################################################################3
    def normalize_column(self, column_name, save_to=None):
        """
        Standardize the specified column and return the standardized data.
        """
        # Create a StandardScaler instance
        scaler = StandardScaler()

        # Standardize the specified column
        self.data[column_name] = scaler.fit_transform(self.data[[column_name]])

        # If a save path is provided, save the standardized data
        if save_to:
            self.data.to_csv(save_to, index=False)

        return self.data
