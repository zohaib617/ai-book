from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class CitationType(Enum):
    ACADEMIC_PAPER = "academic_paper"
    OFFICIAL_DOCUMENTATION = "official_documentation"
    BOOK = "book"
    WEBSITE = "website"

class Citation(BaseModel):
    """
    Model representing a citation for academic content
    """
    id: Optional[str] = None
    type: CitationType
    authors: List[str]
    title: str
    publication: str
    date: str
    url: Optional[str] = None
    accessed_date: Optional[str] = None
    used_in: List[str] = []  # List of chapter/module references

class CitationService:
    """
    Service for managing citations in IEEE/APA format
    """
    def __init__(self):
        self.citations = []

    def add_citation(self, citation: Citation) -> Citation:
        """
        Add a new citation to the system
        """
        if not citation.id:
            # Generate a simple ID (in real implementation, use UUID)
            citation.id = f"cite_{len(self.citations) + 1}"

        self.citations.append(citation)
        return citation

    def format_ieee_citation(self, citation: Citation) -> str:
        """
        Format citation in IEEE style
        """
        authors_str = ", ".join(citation.authors)
        if citation.type == CitationType.ACADEMIC_PAPER:
            return f'{authors_str}, "{citation.title},", {citation.publication}, {citation.date}.'
        elif citation.type == CitationType.OFFICIAL_DOCUMENTATION:
            return f'{authors_str}, "{citation.title}," {citation.publication}, {citation.date}. [Online]. Available: {citation.url}'
        elif citation.type == CitationType.BOOK:
            return f'{authors_str}, {citation.title}. {citation.publication}, {citation.date}.'
        else:  # WEBSITE
            return f'{authors_str}, "{citation.title}," {citation.publication}. [Online]. Available: {citation.url}. [Accessed: {citation.accessed_date}]'

    def format_apa_citation(self, citation: Citation) -> str:
        """
        Format citation in APA style
        """
        authors_str = ", ".join(citation.authors)
        if citation.type == CitationType.ACADEMIC_PAPER:
            return f'{authors_str} ({citation.date}). {citation.title}. {citation.publication}.'
        elif citation.type == CitationType.OFFICIAL_DOCUMENTATION:
            return f'{authors_str} ({citation.date}). {citation.title}. {citation.publication}. Available at: {citation.url}'
        elif citation.type == CitationType.BOOK:
            return f'{authors_str} ({citation.date}). {citation.title}. {citation.publication}.'
        else:  # WEBSITE
            return f'{authors_str} ({citation.date}). {citation.title}. {citation.publication}. Available at: {citation.url} (Accessed: {citation.accessed_date})'

    def get_citations_for_content(self, content_ref: str) -> List[Citation]:
        """
        Get citations relevant to a specific content reference
        """
        relevant_citations = []
        for citation in self.citations:
            if content_ref in citation.used_in:
                relevant_citations.append(citation)
        return relevant_citations

# Create a global instance
citation_service = CitationService()