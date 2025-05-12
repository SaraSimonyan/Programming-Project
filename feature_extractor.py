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
        Extract features from images
        
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
        List of feature vectors
        """
        if not images:
            self.logger.warning("No images provided for feature extraction.")
            return []
            
        if params is None:
            params = {}
            
        self._features = []
        self._feature_names = []
        self.logger.info(f"Extracting features using method: {method}")
        
        for i, img_np in enumerate(images):
            if i % 100 == 0 and i > 0:
                self.logger.info(f"Processed {i}/{len(images)} images")
                
            features = []
            feature_names = []
            
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
                intensity_mean = np.mean(img_gray)
                intensity_std = np.std(img_gray)
                intensity_min = np.min(img_gray)
                intensity_max = np.max(img_gray)
                intensity_median = np.median(img_gray)
                
                features.extend([intensity_mean, intensity_std, intensity_min, intensity_max, intensity_median])
                feature_names.extend(['intensity_mean', 'intensity_std', 'intensity_min', 'intensity_max', 'intensity_median'])
                
                # Shape statistics
                height, width = img_gray.shape
                aspect_ratio = width / height
                
                features.extend([height, width, aspect_ratio])
                feature_names.extend(['height', 'width', 'aspect_ratio'])
                
                # Higher order statistics
                skewness = np.mean(((img_gray - intensity_mean) / (intensity_std + 1e-8)) ** 3)
                kurtosis = np.mean(((img_gray - intensity_mean) / (intensity_std + 1e-8)) ** 4) - 3
                
                features.extend([skewness, kurtosis])
                feature_names.extend(['skewness', 'kurtosis'])
            
            # 2. Histogram features
            if method in ['histogram', 'all']:
                bins = params.get('hist_bins', 32)
                if len(img_np.shape) > 2:  # Color image
                    for channel in range(min(3, img_np.shape[2])):
                        hist, _ = np.histogram(img_np[:, :, channel], bins=bins, range=(0, 256))
                        hist = hist.astype(float) / hist.sum()
                        features.extend(hist)
                        channel_name = ['R', 'G', 'B'][channel]
                        feature_names.extend([f'hist_{channel_name}_{i}' for i in range(bins)])
                else:  # Grayscale
                    hist, _ = np.histogram(img_np, bins=bins, range=(0, 256))
                    hist = hist.astype(float) / hist.sum()
                    features.extend(hist)
                    feature_names.extend([f'hist_gray_{i}' for i in range(bins)])
            
            # 3. Contrast features
            if method in ['contrast', 'all']:
                # Global contrast: standard deviation of pixel values
                global_contrast = np.std(img_gray)
                
                # Local contrast: average of local standard deviations
                local_size = params.get('local_contrast_size', 7)
                if local_size < min(img_gray.shape):
                    local_contrasts = []
                    for i in range(0, img_gray.shape[0] - local_size, local_size):
                        for j in range(0, img_gray.shape[1] - local_size, local_size):
                            patch = img_gray[i:i+local_size, j:j+local_size]
                            local_contrasts.append(np.std(patch))
                    local_contrast = np.mean(local_contrasts) if local_contrasts else 0
                else:
                    local_contrast = global_contrast
                
                features.extend([global_contrast, local_contrast])
                feature_names.extend(['global_contrast', 'local_contrast'])
            
            # 4. Color variance features
            if method in ['color_variance', 'all'] and len(img_np.shape) > 2:
                # Variance of each color channel
                channel_variance = [np.var(img_rgb[:,:,c]) for c in range(3)]
                # Color diversity: variance between channel means
                channel_means = [np.mean(img_rgb[:,:,c]) for c in range(3)]
                color_diversity = np.var(channel_means)
                
                features.extend(channel_variance)
                features.append(color_diversity)
                feature_names.extend(['variance_R', 'variance_G', 'variance_B', 'color_diversity'])
            
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
                
                # Flatten dominant colors and proportions into a feature vector
                features.extend(dominant_colors.flatten())
                features.extend(proportions)
                
                for i in range(num_colors):
                    feature_names.extend([f'dom_color_{i}_R', f'dom_color_{i}_G', f'dom_color_{i}_B'])
                for i in range(num_colors):
                    feature_names.append(f'dom_color_{i}_proportion')
            
            # 6. Edge density features (simple replacement for HOG)
            if method in ['edge_density', 'all']:
                # Simple gradient-based edge detection
                gy, gx = np.gradient(img_gray)
                edge_magnitude = np.sqrt(gx**2 + gy**2)
                
                # Edge density features
                edge_mean = np.mean(edge_magnitude)
                edge_std = np.std(edge_magnitude)
                
                # Threshold-based edge density
                threshold = params.get('edge_threshold', 0.1)
                edge_density = np.mean(edge_magnitude > threshold)
                
                # Directional gradients
                gx_mean, gx_std = np.mean(np.abs(gx)), np.std(np.abs(gx))
                gy_mean, gy_std = np.mean(np.abs(gy)), np.std(np.abs(gy))
                
                # Create edge density vector instead of separate columns
                edge_features = np.array([edge_mean, edge_std, edge_density, 
                                          gx_mean, gx_std, gy_mean, gy_std])
                
                # Add the vector as a single feature
                features.append(edge_features)
                feature_names.append('edge_density_vector')
            
            # Convert features to a numpy array before appending
            self._features.append(np.array(features, dtype=object))
            
            # Only store feature names from the first image
            if i == 0:
                self._feature_names = feature_names
        
        self.logger.info(f"Extracted {len(self._features)} feature vectors with {len(self._features[0]) if self._features else 0} dimensions each")
        print(self._features)
        return self._features
        
    def get_features(self):
        return self._features
    
    def get_feature_names(self):
        return self._feature_names
    
    def to_dataframe(self, image_filenames=None):
        """
        Convert extracted features to a pandas DataFrame
        with one column per scalar or sub‐element, named according
        to self._feature_names (expanding any vector‐features).
        """
        if not self._features:
            self.logger.warning("No features to convert to DataFrame")
            return pd.DataFrame()

        # 1) build flat column names from first feature vector
        flat_cols = []
        first = self._features[0]
        for name, feat in zip(self._feature_names, first):
            if isinstance(feat, np.ndarray) and feat.ndim > 0:
                if name == 'edge_density_vector':
                    flat_cols += [
                        'edge_mean','edge_std','edge_density',
                        'gradient_x_mean','gradient_x_std',
                        'gradient_y_mean','gradient_y_std'
                    ]
                else:
                    flat_cols += [f"{name}_{i}" for i in range(len(feat))]
            else:
                flat_cols.append(name)

        # 2) flatten every feature vector into rows
        rows = []
        for vec in self._features:
            row = []
            for feat in vec:
                if isinstance(feat, np.ndarray) and feat.ndim > 0:
                    row += feat.tolist()
                else:
                    row.append(feat)
            rows.append(row)

        # 2b) verify flat_cols length matches data columns, else generate generic names
        n_cols = len(rows[0])
        if len(flat_cols) != n_cols:
            self.logger.warning(
                f"Column name count ({len(flat_cols)}) != data width ({n_cols}), "
                "using generic feature names."
            )
            flat_cols = [f"feature_{i}" for i in range(n_cols)]

        # 3) build DataFrame
        df = pd.DataFrame(rows, columns=flat_cols)
        if image_filenames and len(image_filenames) == len(rows):
            df.index = image_filenames

        self.logger.info(f"Created DataFrame with shape {df.shape}")
        return df