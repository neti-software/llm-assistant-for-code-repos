from abc import ABC, abstractmethod


class EmbeddingABC(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Generate an embedding for the given text.

        Parameters
        ----------
        text : str
            Input text to embed.

        Returns
        -------
        list[float]
            Vector representation of the input text.
        """
        pass
