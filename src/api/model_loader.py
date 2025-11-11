"""
Model loader for DistilBERT
"""

import logging
from pathlib import Path

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and initialization of the DistilBERT model"""

    def __init__(self, model_path: str = "models/distilbert_exp2"):
        """
        Initialize model loader

        Args:
            model_path: Path to the saved model directory
        """
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.device = None
        self.label_encoder_classes = [
            "Anxiety",
            "Bipolar",
            "Depression",
            "Normal",
            "Stress",
            "Suicidal",
        ]

    def load(self):
        """Load model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}...")

            # Check if model exists
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. "
                    "Please download the model from Colab first!"
                )

            # Determine device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = DistilBertTokenizer.from_pretrained(str(self.model_path))

            # Load model
            logger.info("Loading model...")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                str(self.model_path)
            )
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            logger.info("✅ Model loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise

    def get_model(self):
        """Get the loaded model"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first!")
        return self.model

    def get_tokenizer(self):
        """Get the loaded tokenizer"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first!")
        return self.tokenizer

    def get_device(self):
        """Get the device (CPU/GPU)"""
        return self.device

    def get_classes(self):
        """Get class names"""
        return self.label_encoder_classes
