import logging
import matplotlib.pyplot as plt
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
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
    
    # Extract features
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features(
        images,
        method='basic_stats',
        params={
            'hist_bins': 32,            # Histogram parameters
            'num_dominant_colors': 3,   # Number of dominant colors to extract
            'local_contrast_size': 7,   # Patch size for local contrast
            'edge_threshold': 0.1       # Threshold for edge detection
        }
    )
    
    labels = data_loader.get_labels()
    filenames = data_loader.get_filenames()
    
    print(f"Loaded {len(images)} images with {len(features[0])} features each.")
    
    # Display an image
    data_loader.display_image(2)
    
    # Print feature categories and their dimensions for better understanding
    if len(features) > 0:
        feature_dims = {
            'Basic Stats': 10,  # 5 intensity stats + 3 shape stats + 2 higher order stats
            'Histogram': 32 * 3,  # 32 bins * 3 channels
            'Contrast': 2,  # global and local contrast
            'Color Variance': 4,  # 3 channel variances + color diversity
            'Dominant Colors': 3 * 3 + 3,  # 3 colors (RGB) + 3 proportions
            'Edge Density': 7  # edge_mean, edge_std, edge_density, gx_mean, gx_std, gy_mean, gy_std
        }
        
        print("\nFeature breakdown:")
        for feature_type, dim in feature_dims.items():
            print(f"  - {feature_type}: {dim} dimensions")
            
        # Get the actual feature names for reference
        feature_names = feature_extractor.get_feature_names()
        if feature_names:
            print(f"\nTotal features: {len(feature_names)}")
            
        # Optional: Convert features to DataFrame for easier analysis
        df = feature_extractor.to_dataframe(filenames)
        print(df.head())