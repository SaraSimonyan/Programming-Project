import numpy as np
import logging
from skimage.filters import sobel
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import canny

class FeatureExtractor:
    def __init__(self):
        self._features = []
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging for the FeatureExtractor class"""
        self.logger = logging.getLogger('FeatureExtractor')
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
    def extract_features(self, images, method='all', params=None):
        """
        Extract features from images using various methods
        
        Parameters:
        -----------
        images : list
            List of image arrays to extract features from
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
        if not images:
            self.logger.warning("No images provided for feature extraction.")
            return []
            
        if params is None:
            params = {}
            
        self._features = []
        self.logger.info(f"Extracting features using method: {method}")
        
        for i, img_np in enumerate(images):
            if i % 100 == 0 and i > 0:
                self.logger.info(f"Processed {i}/{len(images)} images")
                
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
        
    def get_features(self):
        return self._features