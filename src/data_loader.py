import numpy as np
from PIL import Image
import logging
import random
from pathlib import Path
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, folder):
        self._folder = folder
        self._images = []
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

    def load_data(self, to_grayscale=False, image_size=(64, 64), max_images=None, extract_labels=False):
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

    def get_images(self):
        return self._images
        
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