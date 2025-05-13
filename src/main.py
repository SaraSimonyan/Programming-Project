import logging
import matplotlib.pyplot as plt
from src.data_loader import DataLoader
from src.feature_extractor import FeatureExtractor
from src.data_analyzer import DataAnalyzer
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    folder = "img/"
    
    # Load the images
    data_loader = DataLoader(folder)
    images = data_loader.load_data(
        to_grayscale=False,  # Keep color for color-based features
        image_size=(64, 64), 
        max_images=100, 
        extract_labels=True
    )
    
    feature_extractor = FeatureExtractor()
    features_df = feature_extractor.extract_features(
        images,
        method='all',
        params={
            'hist_bins': 32,            # Histogram parameters
            'num_dominant_colors': 3,   # Number of dominant colors to extract
            'local_contrast_size': 7,   # Patch size for local contrast
            'edge_threshold': 0.1       # Threshold for edge detection
        }
    )
    
    
    labels = data_loader.get_labels()
    filenames = data_loader.get_filenames()
    
    print(f"Loaded {len(images)} images with {len(features_df.columns)} features each.")
    
    # Display an image
    data_loader.display_image(2)
    
    # display features data
    print("Features DataFrame:")
    print(features_df.columns)
        
    data_analyzer = DataAnalyzer()
    data_analyzer.load_data(features_df)
    data_analyzer.summary_statistics()
    data_analyzer.sort_by_feature('intensity_mean', ascending=False)
    data_analyzer.filter_by_threshold('intensity_mean', 0.5, above=True)