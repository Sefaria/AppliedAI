from langchain_core.documents import Document


class ChunkDocument(Document):
    def __eq__(self, other):
        """Overrides the default implementation of equal check."""
        if isinstance(other, Document):
            return self.page_content == other.page_content and self.metadata == other.metadata
        return False