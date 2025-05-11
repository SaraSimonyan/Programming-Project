from PIL import Image
import numpy as np
import os
from scipy import signal
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from skimage.transform import resize

class DataLoader:
    def __init__(self, folder):
        self._folder = folder
        self._images = []
        self._features = []

    def data_loader(self, to_grayscale = True ):
        image_size = (64, 64)

        for filename in os.listdir(self._folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(self._folder, filename)
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
            - 'all': Combine all feature types (default)
        params : dict
            Additional parameters for specific extraction methods
            
        Returns:
        --------
        List of feature vectors
        """
        if not self._images:
            self.data_loader()
            
        if params is None:
            params = {}
            
        self._features = []
        
        for img_np in self._images:
            features = []
            
            # Make sure image is properly formatted for different extractors
            if len(img_np.shape) == 2:  # If grayscale
                img_gray = img_np
                # Create RGB version if needed
                img_rgb = np.stack((img_np,) * 3, axis=-1) if method in ['all', 'hog'] else None
            else:  # If color
                img_rgb = img_np[:, :, :3]  # Use RGB channels
                img_gray = rgb2gray(img_rgb) if method in ['all', 'lbp', 'edges', 'hog'] else None
            
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
                # Sobel edge detector
                sobel_x = signal.sobel(img_gray, axis=0)
                sobel_y = signal.sobel(img_gray, axis=1)
                magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                
                # Get edge direction histogram
                edge_hist, _ = np.histogram(magnitude, bins=16)
                edge_hist = edge_hist.astype(float) / (edge_hist.sum() + 1e-10)
                features.extend(edge_hist)
            
            self._features.append(np.array(features))
        
        return self._features

    def get_images(self):
     return self._images

    def get_features(self):
        return self._features


