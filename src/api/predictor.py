"""
Predictor class for making predictions with DistilBERT
"""

import logging
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)


class MentalHealthPredictor:
    """Handles predictions using the loaded DistilBERT model"""

    def __init__(self, model_loader):
        """
        Initialize predictor

        Args:
            model_loader: ModelLoader instance with loaded model
        """
        self.model_loader = model_loader
        self.model = model_loader.get_model()
        self.tokenizer = model_loader.get_tokenizer()
        self.device = model_loader.get_device()
        self.classes = model_loader.get_classes()

        # Track predictions for drift detection
        self.prediction_history = []
        self.max_history = 1000  # Keep last 1000 predictions

    def predict(self, text: str) -> Dict:
        """
        Make a single prediction

        Args:
            text: Input text to classify

        Returns:
            Dictionary with prediction results
        """
        try:
            # Tokenize
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Move to device
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()

            # Get class name
            predicted_class = self.classes[predicted_class_idx]

            # Get all probabilities
            probs_dict = {
                class_name: float(probabilities[0][idx].item())
                for idx, class_name in enumerate(self.classes)
            }

            # Store prediction for drift detection
            self._store_prediction(predicted_class_idx, confidence)

            result = {
                "prediction": predicted_class,
                "confidence": float(confidence),
                "probabilities": probs_dict,
                "predicted_class_idx": predicted_class_idx,
            }

            logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.3f})")
            return result

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Make batch predictions

        Args:
            texts: List of input texts

        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting text: {e}")
                results.append(
                    {
                        "error": str(e),
                        "prediction": None,
                        "confidence": 0.0,
                        "probabilities": {},
                    }
                )

        return results

    def _store_prediction(self, class_idx: int, confidence: float):
        """Store prediction for drift detection"""
        self.prediction_history.append(
            {"class_idx": class_idx, "confidence": confidence}
        )

        # Keep only last N predictions
        if len(self.prediction_history) > self.max_history:
            self.prediction_history = self.prediction_history[-self.max_history :]

    def get_prediction_history(self) -> List[Dict]:
        """Get prediction history for drift detection"""
        return self.prediction_history

    def get_prediction_distribution(self) -> Dict[str, float]:
        """Get current prediction distribution"""
        if not self.prediction_history:
            return {class_name: 0.0 for class_name in self.classes}

        # Count predictions per class
        class_counts = {i: 0 for i in range(len(self.classes))}
        for pred in self.prediction_history:
            class_counts[pred["class_idx"]] += 1

        # Convert to distribution
        total = len(self.prediction_history)
        distribution = {
            self.classes[idx]: count / total for idx, count in class_counts.items()
        }

        return distribution
