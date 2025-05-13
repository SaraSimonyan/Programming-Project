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
        
    def get_most_common_dominant_color(self):
        """
        Calculate the average of the most dominant color across the dataset.
        Assumes columns like dom_color_0_R, dom_color_0_G, dom_color_0_B.
        """
        if self._df is None:
            self.logger.warning("No data loaded")
            return None
    

        avg_colors = {
        'color_0': [self._df['dom_color_0_R'].mean(),
                    self._df['dom_color_0_G'].mean(),
                    self._df['dom_color_0_B'].mean()],
        'color_1': [self._df['dom_color_1_R'].mean(),
                    self._df['dom_color_1_G'].mean(),
                    self._df['dom_color_1_B'].mean()],
        'color_2': [self._df['dom_color_2_R'].mean(),
                    self._df['dom_color_2_G'].mean(),
                    self._df['dom_color_2_B'].mean()]
        }

        # Create the DataFrame
        avg_colors_df = pd.DataFrame.from_dict(avg_colors, orient='index', columns=['red', 'green', 'blue'])
      
        return avg_colors_df
    
    def mean_diff_by_feature(self, feature):
        """
        Create a new column in the DataFrame containing the difference between
        each value in the specified feature column and its mean.

        Parameters:
        -----------
        feature : str
            The name of the column for which to calculate mean difference.

        Returns:
        --------
        DataFrame
            A copy of the DataFrame with an additional column showing the
            difference from the mean for the specified feature.
        """
        if self._df is None:
            self.logger.warning("No data loaded")
            return None

        if feature not in self._df.columns:
            self.logger.warning(f"Feature '{feature}' not found in DataFrame")
            return None

        feature_mean = self._df[feature].mean()
        column_name = f"{feature} difference"

        self._df_copy = self._df.copy()
        self._df_copy[column_name] = self._df[feature] - feature_mean

        return self._df_copy

