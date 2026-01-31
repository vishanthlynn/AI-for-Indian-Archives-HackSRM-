import cv2
import numpy as np
from PIL import Image

class ImagePreprocessor:
    def __init__(self):
        pass

    def load_image(self, image_path):
        """Loads an image from path or PIL Image."""
        if isinstance(image_path, str):
            # Load with OpenCV (BGR)
            return cv2.imread(image_path)
        elif isinstance(image_path, Image.Image):
            # Convert PIL RGB to OpenCV BGR
            return cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
        elif isinstance(image_path, np.ndarray):
            return image_path
        else:
            raise ValueError("Unsupported image format")

    def to_grayscale(self, image):
        """Converts BGR image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def apply_clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Applies Contrast Limited Adaptive Histogram Equalization."""
        gray = self.to_grayscale(image)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray)

    def denoise(self, image, h=10):
        """Applies Non-Local Means Denoising."""
        # Check if color or gray
        if len(image.shape) == 3:
             return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
        else:
             return cv2.fastNlMeansDenoising(image, None, h, 7, 21)

    def adapt_threshold(self, image, block_size=11, C=2):
        """Applies adaptive thresholding for binarization."""
        gray = self.to_grayscale(image)
        # Adaptive Gaussian Thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, C
        )
        return binary

    def deskew(self, image):
        """Robust deskewing based on text orientation."""
        gray = self.to_grayscale(image)
        
        # Check if background is dark
        # If median pixel is low (<127), it's likely dark background
        if np.median(gray) < 127:
            # excessive white text on black.
            # No inversion needed to find white text
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        else:
            # Standard: Black text on white. Invert to find text.
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        
        # If no text found, return original
        if len(coords) < 10:
            return image, 0.0

        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # Optimization: Heritage docs are rarely skewed > 10-15 degrees.
        # If angle is large (> 45 or near 90), it's likely measuring the page border, not text lines.
        # Let's dampen extreme rotations which confuse OCR.
        if abs(angle) > 45:
             angle = 0
             
        # Rotate the image to deskew it
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated, angle

    def process_pipeline(self, image_input):
        """Full pipeline: Denoise -> Deskew -> CLAHE -> Binarization"""
        img = self.load_image(image_input)
        
        # 1. Denoise (Input is usually color/grey)
        denoised = self.denoise(img)
        
        # 2. Deskew
        deskewed, _ = self.deskew(denoised)
        
        # 3. Enhance Contrast (CLAHE works on Grayscale)
        enhanced = self.apply_clahe(deskewed)
        
        # 4. Binarize (Optional, depending on if OCR expects binary or gray)
        # Doctr works well with RGB/Gray, but for extraction we might want binary.
        # Let's keep Enhanced Grayscale as primary output for OCR, Binary for visualization.
        binary = self.adapt_threshold(enhanced)
        
        return {
            "original": img,
            "denoised": denoised,
            "deskewed": deskewed,
            "enhanced": enhanced, # Best for OCR usually
            "binary": binary # Good for layout analysis
        }
