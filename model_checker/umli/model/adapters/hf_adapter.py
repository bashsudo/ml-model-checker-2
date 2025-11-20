"""
HuggingFaceAdapter - Adapter for HuggingFace Transformers models.
"""

from typing import Any, Optional

from transformers import AutoModel, AutoTokenizer

from model_checker.umli.model.unified_model import BaseModelAdapter


class HuggingFaceAdapter(BaseModelAdapter):
    """
    Adapter for HuggingFace Transformers models.
    """

    def __init__(
        self,
        backend_model: Any,
        tokenizer: Optional[Any] = None,
        config: Optional[dict] = None,
    ):
        """
        Initialize HuggingFace adapter.

        Args:
            backend_model: HuggingFace model instance
            tokenizer: Optional tokenizer for text inputs
            config: Optional configuration dictionary
        """
        super().__init__(backend_model, config)
        self.tokenizer = tokenizer

    def forward(self, batch: Any) -> Any:
        """
        Forward pass through the model.

        Args:
            batch: Input data (text or tokenized inputs)

        Returns:
            Model output
        """
        if self.tokenizer and isinstance(batch, (str, list)):
            # Tokenize text inputs
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            )
            return self.backend_model(**inputs)
        else:
            return self.backend_model(batch)

    def backward(self, loss: Any) -> None:
        """
        Backward pass for HuggingFace models.
        Typically handled by Trainer class.
        """
        if hasattr(loss, "backward"):
            loss.backward()

    def step(self) -> None:
        """Step is handled by HuggingFace Trainer."""

    def fit(self, dataset: Any) -> None:
        """
        Fit is typically handled by HuggingFace Trainer.
        This is a placeholder.
        """
        raise NotImplementedError(
            "HuggingFace models typically use Trainer class. Use Trainer.train() instead."
        )

    def predict(self, batch: Any) -> Any:
        """
        Generate predictions.

        Args:
            batch: Input data

        Returns:
            Predictions
        """
        self.backend_model.eval()
        with (
            self.backend_model.no_grad()
            if hasattr(self.backend_model, "no_grad")
            else __import__("contextlib").nullcontext()
        ):
            return self.forward(batch)

    def save(self, path: str) -> None:
        """Save HuggingFace model."""
        self.backend_model.save_pretrained(path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        """Load HuggingFace model."""
        self.backend_model = AutoModel.from_pretrained(path)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        except Exception:
            pass  # Tokenizer may not be available
