import pandas as pd
import logging

class DataAnalyzer:
    def __init__(self):
        self._df = None  # Internal storage for the DataFrame
        self._setup_logging()  # Initialize logging

    def get_df(self):
        """Return the currently loaded DataFrame."""
        return self._df

    def _setup_logging(self):
        """Set up a logger for the DataAnalyzer class."""
        self.logger = logging.getLogger('DataAnalyzer')
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def load_data(self, df):
        """
        Load a DataFrame for analysis.
        Parameters:
        - df: A pandas DataFrame to be stored and analyzed.
        """
        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input must be a pandas DataFrame")
            raise ValueError("Input must be a pandas DataFrame")
        self._df = df
        self.logger.info(f"Data loaded with shape {df.shape}")

    def summary_statistics(self):
        """
        Return basic descriptive statistics (mean, std, min, max, etc.)
        for numeric columns in the DataFrame.
        """
        if self._df is None:
            self.logger.warning("No data loaded")
            return None
        return self._df.describe()

    def sort_by_feature(self, feature, ascending=True):
        """
        Sort the DataFrame by a specific feature.
        Parameters:
        - feature: column name to sort by
        - ascending: sort direction (True for ascending, False for descending)
        """
        if self._df is None:
            self.logger.warning("No data loaded")
            return None
        if feature not in self._df.columns:
            self.logger.warning(f"Feature '{feature}' not found in columns")
            return None
        return self._df.sort_values(by=feature, ascending=ascending)

    def filter_by_threshold(self, feature, threshold, above=True):
        """
        Filter rows based on whether a feature's value is above or below a threshold.
        Parameters:
        - feature: column name to filter on
        - threshold: numeric value to compare against
        - above: if True, returns rows with value > threshold; else <= threshold
        """
        if self._df is None:
            self.logger.warning("No data loaded")
            return None
        if feature not in self._df.columns:
            self.logger.warning(f"Feature '{feature}' not found in columns")
            return None
        if above:
            return self._df[self._df[feature] > threshold]
        else:
            return self._df[self._df[feature] <= threshold]
