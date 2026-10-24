class VectorStoreError(Exception):
    """Base class for vector store failures."""


class VectorStoreUnavailable(VectorStoreError):
    """Qdrant could not be reached, timed out, or rejected the request."""


class CollectionMismatch(VectorStoreError):
    """The existing collection was created for vectors of a different size or metric."""
