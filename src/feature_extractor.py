import numpy as np
import logging
import pandas as pd
from sklearn.cluster import KMeans

class FeatureExtractor:
    def __init__(self):
        self._features = []
        self._feature_names = []
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
    
    def _rgb_to_gray(self, rgb_img):
        """Convert RGB image to grayscale using standard coefficients"""
        return np.dot(rgb_img[..., :3], [0.2989, 0.5870, 0.1140])
    
    def extract_features(self, images, method='all', params=None):
        """
        Extract features from images and return them as a DataFrame
        
        Parameters:
        -----------
        images : list
            List of image arrays to extract features from
        method : str
            The feature extraction method to use:
            - 'histogram': Color/intensity histogram
            - 'basic_stats': Basic statistics like mean, std, etc.
            - 'contrast': Image contrast features
            - 'color_variance': Color variance features
            - 'dominant_colors': Dominant colors in the image
            - 'edge_density': Simple edge detection and density calculations
            - 'all': Combine all feature types (default)
        params : dict
            Additional parameters for specific extraction methods
                
        Returns:
        --------
        DataFrame with image features (one row per image)
        """
        if not images:
            self.logger.warning("No images provided for feature extraction.")
            return pd.DataFrame()
            
        if params is None:
            params = {}
        
        self.logger.info(f"Extracting features using method: {method}")
        
        # Process the first image to determine all feature names
        first_features = self._extract_image_features(images[0], method, params)
        
        # Create DataFrame with the right columns and pre-allocate rows
        feature_names = list(first_features.keys())
        self._features = pd.DataFrame(columns=feature_names, index=range(len(images)), dtype=float)
        
        # Set the first row with already extracted features
        self._features.iloc[0] = list(first_features.values())
        
        # Process the rest of the images
        for i in range(1, len(images)):
            if i % 100 == 0:
                self.logger.info(f"Processed {i}/{len(images)} images")
                
            features = self._extract_image_features(images[i], method, params)
            self._features.iloc[i] = list(features.values())
        
        # Store feature names for reference
        self._feature_names = feature_names
        
        self.logger.info(f"Extracted features for {len(self._features)} images with {len(feature_names)} dimensions each")
        return self._features

    def _extract_image_features(self, img_np, method, params):
        """Extract features from a single image"""
        features = {}  # Use a dictionary to store features with their names
        
        if len(img_np.shape) == 2:  # If grayscale
            img_gray = img_np
            # Create RGB version if needed
            img_rgb = np.stack((img_np,) * 3, axis=-1)
        else:  # If color
            img_rgb = img_np[:, :, :3]  # Use RGB channels
            img_gray = self._rgb_to_gray(img_rgb)
        
        # 1. Basic statistical features
        if method in ['basic_stats', 'all']:
            # Intensity statistics
            features['intensity_mean'] = float(np.mean(img_gray))
            features['intensity_std'] = float(np.std(img_gray))
            features['intensity_min'] = float(np.min(img_gray))
            features['intensity_max'] = float(np.max(img_gray))
            features['intensity_median'] = float(np.median(img_gray))
            
            # Shape statistics
            height, width = img_gray.shape
            features['height'] = float(height)
            features['width'] = float(width)
            features['aspect_ratio'] = float(width / height)
            
            # Higher order statistics
            intensity_mean = features['intensity_mean']
            intensity_std = features['intensity_std']
            features['skewness'] = float(np.mean(((img_gray - intensity_mean) / (intensity_std + 1e-8)) ** 3))
            features['kurtosis'] = float(np.mean(((img_gray - intensity_mean) / (intensity_std + 1e-8)) ** 4) - 3)
        
        # 2. Histogram features
        if method in ['histogram', 'all']:
            bins = params.get('hist_bins', 32)
            if len(img_np.shape) > 2:  # Color image
                for channel in range(min(3, img_np.shape[2])):
                    hist, _ = np.histogram(img_np[:, :, channel], bins=bins, range=(0, 256))
                    hist = hist.astype(float) / hist.sum()
                    channel_name = ['R', 'G', 'B'][channel]
                    for bin_idx, value in enumerate(hist):
                        features[f'hist_{channel_name}_{bin_idx}'] = float(value)
            else:  # Grayscale
                hist, _ = np.histogram(img_np, bins=bins, range=(0, 256))
                hist = hist.astype(float) / hist.sum()
                for bin_idx, value in enumerate(hist):
                    features[f'hist_gray_{bin_idx}'] = float(value)
        
        # 3. Contrast features
        if method in ['contrast', 'all']:
            # Global contrast: standard deviation of pixel values
            features['global_contrast'] = float(np.std(img_gray))
            
            # Local contrast: average of local standard deviations
            local_size = params.get('local_contrast_size', 7)
            if local_size < min(img_gray.shape):
                local_contrasts = []
                for i_local in range(0, img_gray.shape[0] - local_size, local_size):
                    for j in range(0, img_gray.shape[1] - local_size, local_size):
                        patch = img_gray[i_local:i_local+local_size, j:j+local_size]
                        local_contrasts.append(np.std(patch))
                features['local_contrast'] = float(np.mean(local_contrasts) if local_contrasts else 0)
            else:
                features['local_contrast'] = features['global_contrast']
        
        # 4. Color variance features
        if method in ['color_variance', 'all'] and len(img_np.shape) > 2:
            # Variance of each color channel
            features['variance_R'] = float(np.var(img_rgb[:,:,0]))
            features['variance_G'] = float(np.var(img_rgb[:,:,1]))
            features['variance_B'] = float(np.var(img_rgb[:,:,2]))
            
            # Color diversity: variance between channel means
            channel_means = [np.mean(img_rgb[:,:,c]) for c in range(3)]
            features['color_diversity'] = float(np.var(channel_means))
        
        # 5. Dominant colors
        if method in ['dominant_colors', 'all'] and len(img_np.shape) > 2:
            num_colors = params.get('num_dominant_colors', 3)

            # Reshape image to be a list of pixels
            pixels = img_rgb.reshape(-1, 3)

            # Sample pixels for faster processing
            sample_size = min(1000, pixels.shape[0])
            sampled_pixels = pixels[np.random.choice(pixels.shape[0], sample_size, replace=False)]
            
            # Use K-means to find dominant colors
            kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init=10)
            kmeans.fit(sampled_pixels)
            
            # Get colors and their proportions
            dominant_colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            counts = np.bincount(labels)
            proportions = counts / sample_size
            
            # Add each dominant color and proportion as separate columns
            for color_idx in range(num_colors):
                features[f'dom_color_{color_idx}_R'] = float(dominant_colors[color_idx, 0])
                features[f'dom_color_{color_idx}_G'] = float(dominant_colors[color_idx, 1])
                features[f'dom_color_{color_idx}_B'] = float(dominant_colors[color_idx, 2])
                features[f'dom_color_{color_idx}_proportion'] = float(proportions[color_idx])
        
        # 6. Edge density features (simple replacement for HOG)
        if method in ['edge_density', 'all']:
            # Simple gradient-based edge detection
            gy, gx = np.gradient(img_gray)
            edge_magnitude = np.sqrt(gx**2 + gy**2)
            
            # Edge density features
            features['edge_mean'] = float(np.mean(edge_magnitude))
            features['edge_std'] = float(np.std(edge_magnitude))
            
            # Threshold-based edge density
            threshold = params.get('edge_threshold', 0.1)
            features['edge_density'] = float(np.mean(edge_magnitude > threshold))
            
            # Directional gradients
            features['gradient_x_mean'] = float(np.mean(np.abs(gx)))
            features['gradient_x_std'] = float(np.std(np.abs(gx)))
            features['gradient_y_mean'] = float(np.mean(np.abs(gy)))
            features['gradient_y_std'] = float(np.std(np.abs(gy)))
        
        return features

