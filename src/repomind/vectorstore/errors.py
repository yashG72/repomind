class VectorStoreError(Exception):
    """Base class for vector store failures."""


class VectorStoreUnavailable(VectorStoreError):
    """Qdrant could not be reached, timed out, or rejected the request."""


class CollectionMismatch(VectorStoreError):
    """The existing collection was created for vectors of a different size or metric."""


class KeywordSearchUnavailable(VectorStoreError):
    """The collection was created before keyword search existed (it has no keyword index)."""
