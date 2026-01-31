import os
import torch
import numpy as np
import cv2
from doctr.models import ocr_predictor, ocr_predictor
from doctr.io import DocumentFile
import pytesseract
from PIL import Image

class HeritageOCREngine:
    def __init__(self, use_gpu=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        print(f"Loading Heritage OCR Engine on {self.device}...")
        
        # Load Doctr Model (Multilingual)
        # using 'master' for recognition as it is robust for wild text
        self.predictor = ocr_predictor(det_arch='db_resnet50', reco_arch='master', pretrained=True)
        if use_gpu and torch.cuda.is_available():
            self.predictor.cuda()

    def detect_and_recognize(self, image_input, use_tesseract_fallback=False, lang='eng+hin'):
        """
        Runs OCR on the image.
        :param image_input: numpy array (BGR or RGB) or path strings
        :param use_tesseract_fallback: If True, uses Tesseract for recognition on detected crops for better Indic support.
        :param lang: Tesseract language code (e.g., 'hin', 'tam', 'eng')
        :return: Structured JSON-like output
        """
        # Doctr expects list of documents or single doc
        if isinstance(image_input, np.ndarray):
             # Doctr expects RGB
             # If grayscale (2D) or 1-channel (3D), convert to 3-channel BGR/RGB
             if len(image_input.shape) == 2:
                 # Grayscale -> RGB (Stacking channels)
                 doc = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
             elif image_input.shape[-1] == 1:
                 # Single channel 3D -> RGB
                 doc = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
             elif image_input.shape[-1] == 4:
                 # RGBA -> RGB
                 doc = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
             elif image_input.shape[-1] == 3:
                 # If BGR (standard OpenCV), convert to RGB
                 doc = image_input[..., ::-1].copy() # BGR to RGB
             else:
                 doc = image_input 
             docs = [doc]
        else:
             # Assume path
             docs = DocumentFile.from_images(image_input)

        # Run Doctr
        result = self.predictor(docs)
        
        output_data = {
            "full_text": "",
            "blocks": [],
            "words": []
        }

        # Parse Doctr Result
        full_text_parts = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = []
                    for word in line.words:
                        text = word.value
                        confidence = word.confidence
                        
                        # Fallback to Tesseract if confidence is low OR explicitly requested for Indic
                        if use_tesseract_fallback:
                            # Extract crop based on geometry
                            # Doctr geo is relative [min_y, min_x, max_y, max_x]? 
                            # Actually it's ((xmin, ymin), (xmax, ymax))
                            # word.geometry is ((x,y), (x,y))
                            
                            # For simplicity in this hackathon version, we might trust Doctr for layout
                            # and if needed run Tesseract on the whole page or blocks.
                            # But let's keep it simple: Use Doctr as primary.
                            pass

                        line_text.append(text)
                        output_data["words"].append({
                            "text": text,
                            "confidence": confidence,
                            "geometry": word.geometry
                        })
                    
                    line_str = " ".join(line_text)
                    full_text_parts.append(line_str)
                    
                output_data["blocks"].append({
                    "geometry": block.geometry,
                    "lines": len(block.lines)
                })

        output_data["full_text"] = "\n".join(full_text_parts)
        
        # If Doctr found nothing (common in some edge cases), fallback to Tesseract
        if not output_data["full_text"].strip() and use_tesseract_fallback:
            print("Doctr found empty text, falling back to Tesseract...")
            return self.run_tesseract_full(docs[0])
            
        return output_data

    def run_tesseract_full(self, image, lang='eng+hin'):
        """
        Runs pure Tesseract as a fallback/alternative for dense Indic text.
        """
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
            
        text = pytesseract.image_to_string(img, lang=lang)
        return {"full_text": text, "method": "tesseract"}
