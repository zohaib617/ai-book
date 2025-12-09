import os
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid

class Diagram(BaseModel):
    """
    Model representing a diagram for the robotics book
    """
    id: str
    title: str
    alt_text: str
    file_path: str
    caption: str
    module: str
    created_at: datetime = None
    updated_at: datetime = None

class DiagramService:
    """
    Service for managing diagrams and images with ALT-text requirements
    """
    def __init__(self, static_dir: str = "docs/static/img"):
        self.static_dir = static_dir
        # Create directory if it doesn't exist
        os.makedirs(static_dir, exist_ok=True)
        self.diagrams = []

    def upload_diagram(self, title: str, alt_text: str, caption: str, module: str,
                      file_content: bytes, filename: str) -> Diagram:
        """
        Upload a diagram with proper ALT-text and metadata
        """
        # Generate unique ID
        diagram_id = str(uuid.uuid4())

        # Create file path
        file_path = os.path.join(self.static_dir, f"{diagram_id}_{filename}")

        # Save the file
        with open(file_path, 'wb') as f:
            f.write(file_content)

        # Create diagram object
        diagram = Diagram(
            id=diagram_id,
            title=title,
            alt_text=alt_text,
            file_path=file_path,
            caption=caption,
            module=module,
            created_at=datetime.now()
        )

        self.diagrams.append(diagram)
        return diagram

    def get_diagram_by_id(self, diagram_id: str) -> Optional[Diagram]:
        """
        Retrieve a diagram by its ID
        """
        for diagram in self.diagrams:
            if diagram.id == diagram_id:
                return diagram
        return None

    def get_diagrams_by_module(self, module: str) -> List[Diagram]:
        """
        Get all diagrams for a specific module
        """
        return [d for d in self.diagrams if d.module == module]

    def validate_alt_text(self, alt_text: str) -> bool:
        """
        Validate that ALT text is descriptive and appropriate length
        """
        if not alt_text or len(alt_text.strip()) == 0:
            return False

        # Check for minimum length (meaningful description)
        if len(alt_text.strip()) < 10:
            return False

        # Check for common non-descriptive alt text
        non_descriptive = ["image", "diagram", "picture", "graph", "chart"]
        if alt_text.lower().strip() in non_descriptive:
            return False

        return True

    def update_diagram(self, diagram_id: str, title: Optional[str] = None,
                      alt_text: Optional[str] = None, caption: Optional[str] = None) -> Optional[Diagram]:
        """
        Update diagram information
        """
        for i, diagram in enumerate(self.diagrams):
            if diagram.id == diagram_id:
                if title is not None:
                    diagram.title = title
                if alt_text is not None:
                    if self.validate_alt_text(alt_text):
                        diagram.alt_text = alt_text
                    else:
                        raise ValueError("ALT text must be descriptive and at least 10 characters")
                if caption is not None:
                    diagram.caption = caption
                diagram.updated_at = datetime.now()

                # Update the diagram in the list
                self.diagrams[i] = diagram
                return diagram

        return None

# Create a global instance
diagram_service = DiagramService()