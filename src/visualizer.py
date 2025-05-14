# Simulate a sample dataset
class Visualizer:
    def __init__(self,dataframe):
        self._df = dataframe.get_df()

#  DataAnalyzer-style operations 

# 1. Summary statistics
        self._summary_stats = self._df.describe()

# 2. Sort by edge_density
        self._sorted_df = self._df.sort_values(by='edge_density')

# 3. Filter by edge_density > 0.5
        self._filtered_df = self._df[self._df['edge_density'] > 0.5]

# 4. Average dominant color (Color 0)
        self._avg_color = {
            'red': self._df['dom_color_0_R'].mean(),
            'green': self._df['dom_color_0_G'].mean(),
            'blue': self._df['dom_color_0_B'].mean()
        }

# 5. Mean difference column for intensity_mean
        self._mean_diff_df = self._df.copy()
        self._mean_diff_df['intensity_mean_diff'] = self._df['intensity_mean'] - self._df['intensity_mean'].mean()

# Plots

# 1. Histogram of intensity_mean
    def hist_plot(self):
        plt.figure()
        plt.hist(self._df['intensity_mean'], bins=10, edgecolor='black')
        plt.title('Histogram of Intensity Mean')
        plt.xlabel('Intensity Mean')
        plt.ylabel('Frequency')
        plt.show()
        return
# 2. Scatter plot of intensity_mean vs intensity_std
    def scatter_plot(self):
        plt.figure()
        plt.scatter(self._df['intensity_mean'], self._df['intensity_std'], alpha=0.7)
        plt.title('Scatter Plot: Intensity Mean vs Std')
        plt.xlabel('Intensity Mean')
        plt.ylabel('Intensity Std')
        plt.show()
        return
# 3. Box plot of dominant color channels
    def box_plot(self):
        plt.figure()
        plt.boxplot([self._df['dom_color_0_R'], self._df['dom_color_0_G'], self._df['dom_color_0_B']], labels=['Red', 'Green', 'Blue'])
        plt.title('Box Plot of Dominant Color Channels')
        plt.show()
        return
# 4. Line plot of intensity_mean difference
    def line_plot(self):
        plt.figure()
        plt.plot(self._mean_diff_df['intensity_mean_diff'], marker='o')
        plt.title('Line Plot of Intensity Mean Difference from Mean')
        plt.xlabel('Sample Index')
        plt.ylabel('Difference from Mean')
        plt.show()
        return
# 5. Bar chart of average RGB values
    def bar_chart(self):
        plt.figure()
        plt.bar(['Red', 'Green', 'Blue'], [self._avg_color['red'], self._avg_color['green'], self._avg_color['blue']])
        plt.title('Bar Chart of Average Dominant Color (Color 0)')
        plt.ylabel('Average Value')
        plt.show()
        return
    
    
    def plotting(self):
        self.hist_plot()
        self.scatter_plot()
        self.box_plot()
        self.line_plot()
        self.bar_chart()
