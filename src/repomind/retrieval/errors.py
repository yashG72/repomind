class RepositoryNotIndexed(Exception):
    """A search was made for a repository (or commit) that has not been indexed."""


class InvalidQuery(ValueError):
    """The search query is empty or otherwise unusable."""
