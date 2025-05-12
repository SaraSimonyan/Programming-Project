import logging
import matplotlib.pyplot as plt
from data_loader import DataLoader
from feature_extractor import FeatureExtractor

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
        to_grayscale=True, 
        image_size=(64, 64), 
        max_images=100, 
        extract_labels=True
    )
    
    # Extract features
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features(
        images,
        method='all', 
        params={'pixels_per_cell': (8, 8), 'lbp_radius': 3}
    )
    
    labels = data_loader.get_labels()
    filenames = data_loader.get_filenames()
    
    print(f"Loaded {len(images)} images with {len(features[0])} features each.")
    
    # Display an image
    data_loader.display_image(2)