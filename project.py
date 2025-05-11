from PIL import Image
import numpy as np
import os
from skimage.filters import sobel
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import canny
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import logging
import random
from pathlib import Path
import pickle

class DataLoader:
    def __init__(self, folder):
        self._folder = folder
        self._images = []
        self._features = []
        self._labels = []
        self._filenames = []
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging for the DataLoader class"""
        self.logger = logging.getLogger('DataLoader')
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def data_loader(self, to_grayscale=True, image_size=(64, 64), max_images=None, extract_labels=False):
        """
        Load images from the specified folder
        
        Parameters:
        -----------
        to_grayscale : bool
            Whether to convert images to grayscale
        image_size : tuple
            Size to resize images to
        max_images : int or None
            Maximum number of images to load, or None for all
        extract_labels : bool
            Whether to extract labels from directory structure
            
        Returns:
        --------
        List of images as numpy arrays
        """
        self._images = []
        self._filenames = []
        self._labels = []
        
        folder_path = Path(self._folder)
        image_files = []
        
        # Find all image files
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            image_files.extend(list(folder_path.glob(f'**/*{ext}')))
            image_files.extend(list(folder_path.glob(f'**/*{ext.upper()}')))
        
        # Sample if max_images is specified
        if max_images is not None and max_images < len(image_files):
            image_files = random.sample(image_files, max_images)
            
        self.logger.info(f"Loading {len(image_files)} images from {self._folder}")
        
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                img = img.resize(image_size)
                img = img.convert('RGBA')
                
                img_np = np.array(img)
                if to_grayscale:
                    r = img_np[:, :, 0]
                    g = img_np[:, :, 1]
                    b = img_np[:, :, 2]
                    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                    img_np = gray.astype(np.uint8)
                
                self._images.append(img_np)
                self._filenames.append(str(img_path))
                
                # Extract label from parent directory name if requested
                if extract_labels:
                    label = img_path.parent.name
                    self._labels.append(label)
                    
            except Exception as e:
                self.logger.error(f"Error loading image {img_path}: {e}")
                
        self.logger.info(f"Successfully loaded {len(self._images)} images")
        return self._images

    def extract_features(self, method='all', params=None):
        """
        Extract features from loaded images using various methods
        
        Parameters:
        -----------
        method : str
            The feature extraction method to use:
            - 'flatten': simple flattening of the image
            - 'hog': Histogram of Oriented Gradients
            - 'lbp': Local Binary Pattern
            - 'histogram': Color/intensity histogram
            - 'edges': Edge detection features
            - 'canny': Canny edge detector features
            - 'all': Combine all feature types (default)
        params : dict
            Additional parameters for specific extraction methods
            
        Returns:
        --------
        List of feature vectors
        """
        if not self._images:
            self.logger.warning("No images loaded. Loading images first...")
            self.data_loader()
            
        if params is None:
            params = {}
            
        self._features = []
        self.logger.info(f"Extracting features using method: {method}")
        
        for i, img_np in enumerate(self._images):
            if i % 100 == 0 and i > 0:
                self.logger.info(f"Processed {i}/{len(self._images)} images")
                
            features = []
            
            # Make sure image is properly formatted for different extractors
            if len(img_np.shape) == 2:  # If grayscale
                img_gray = img_np
                # Create RGB version if needed
                img_rgb = np.stack((img_np,) * 3, axis=-1) if method in ['all', 'hog'] else None
            else:  # If color
                img_rgb = img_np[:, :, :3]  # Use RGB channels
                img_gray = rgb2gray(img_rgb) if method in ['all', 'lbp', 'edges', 'hog', 'canny'] else None
            
            # 1. Flatten features (simplest approach)
            if method in ['flatten', 'all']:
                if 'subsample' in params:
                    subsample = params['subsample']
                    flattened = resize(img_gray, (subsample, subsample)).flatten()
                else:
                    flattened = img_gray.flatten()
                features.extend(flattened)
            
            # 2. HOG features
            if method in ['hog', 'all']:
                pixels_per_cell = params.get('pixels_per_cell', (8, 8))
                orientations = params.get('orientations', 9)
                
                # Ensure img is properly sized for HOG
                img_for_hog = resize(img_gray, (64, 64)) if img_gray.shape[0] != 64 else img_gray
                
                hog_features = hog(
                    img_for_hog, 
                    orientations=orientations,
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=(2, 2),
                    visualize=False,
                    feature_vector=True
                )
                features.extend(hog_features)
            
            # 3. LBP features
            if method in ['lbp', 'all']:
                radius = params.get('lbp_radius', 3)
                n_points = params.get('lbp_points', 8 * radius)
                
                lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
                # Histogram of LBP values
                n_bins = params.get('lbp_bins', n_points + 2)
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
                lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
                features.extend(lbp_hist)
            
            # 4. Histogram features
            if method in ['histogram', 'all']:
                bins = params.get('hist_bins', 32)
                if len(img_np.shape) > 2:  # Color image
                    for channel in range(min(3, img_np.shape[2])):
                        hist, _ = np.histogram(img_np[:, :, channel], bins=bins, range=(0, 256))
                        hist = hist.astype(float) / hist.sum()
                        features.extend(hist)
                else:  # Grayscale
                    hist, _ = np.histogram(img_np, bins=bins, range=(0, 256))
                    hist = hist.astype(float) / hist.sum()
                    features.extend(hist)
            
            # 5. Edge features
            if method in ['edges', 'all']:
                sobel_x = sobel(img_gray, axis=0)
                sobel_y = sobel(img_gray, axis=1)
                magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                
                # Get edge direction histogram
                edge_hist, _ = np.histogram(magnitude, bins=16)
                edge_hist = edge_hist.astype(float) / (edge_hist.sum() + 1e-10)
                features.extend(edge_hist)
                
            # 6. Canny edge detector features
            if method in ['canny', 'all']:
                sigma = params.get('canny_sigma', 1.0)
                edges = canny(img_gray, sigma=sigma)
                # Percentage of edge pixels
                edge_density = np.sum(edges) / edges.size
                # Edge histogram by regions
                regions = 4
                h, w = edges.shape
                region_stats = []
                for i in range(regions):
                    for j in range(regions):
                        region = edges[i*h//regions:(i+1)*h//regions, 
                                       j*w//regions:(j+1)*w//regions]
                        region_stats.append(np.sum(region) / region.size)
                features.append(edge_density)
                features.extend(region_stats)
            
            self._features.append(np.array(features))
        
        self.logger.info(f"Extracted {len(self._features)} feature vectors with {len(self._features[0])} dimensions each")
        return self._features

    def get_images(self):
        return self._images

    def get_features(self):
        return self._features
        
    def get_labels(self):
        return self._labels
        
    def get_filenames(self):
        return self._filenames
    
    def display_image(self, index):
        """
        Display an image at the specified index
        
        Parameters:
        -----------
        index : int
            Index of the image to display
        """
        if 0 <= index < len(self._images):
            plt.imshow(self._images[index])
            plt.axis('off')
            plt.show()
        else:
            self.logger.error(f"Index {index} out of range. Cannot display image.")
    

if __name__ == "__main__":
    folder = "img/"
    data_loader = DataLoader(folder)
    
    # Load images
    images = data_loader.data_loader(to_grayscale=True, image_size=(64, 64), max_images=100, extract_labels=True)
    
    # Extract features
    features = data_loader.extract_features(method='all', params={'pixels_per_cell': (8, 8), 'lbp_radius': 3})
    
    # Get labels and filenames
    labels = data_loader.get_labels()
    filenames = data_loader.get_filenames()
    
    print(f"Loaded {len(images)} images with {len(features[0])} features each.")
    
    # Display an image
    data_loader.display_image(2)
    
    