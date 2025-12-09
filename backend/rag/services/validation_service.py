from typing import List, Dict, Any
import re
from pydantic import BaseModel

class ValidationResult(BaseModel):
    is_valid: bool
    issues: List[str]
    suggestions: List[str]

class ContentValidationService:
    """
    Service for validating content accuracy and compliance with project standards
    """
    def __init__(self):
        # Common robotics/ROS terms that should be used correctly
        self.technical_terms = {
            "ROS 2", "ROS", "Node", "Topic", "Service", "rclpy", "URDF",
            "Gazebo", "Isaac", "VSLAM", "Nav2", "LiDAR", "IMU", "SLAM",
            "Qdrant", "OpenAI", "RAG", "VLA", "Whisper", "LLM"
        }

        # Regex patterns for detecting potential issues
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "url": r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            "citation_placeholder": r'\[CITATION\]|\[REFERENCE\]|\[SOURCE\]'
        }

    def validate_content(self, content: str, source_module: str, source_chapter: str) -> ValidationResult:
        """
        Validate content for technical accuracy and compliance
        """
        issues = []
        suggestions = []

        # Check for potential hallucinations (unverified claims)
        hallucination_issues = self._check_for_hallucinations(content)
        issues.extend(hallucination_issues)

        # Check for proper technical terminology
        terminology_issues = self._check_technical_terminology(content)
        issues.extend(terminology_issues)

        # Check for citation placeholders
        citation_issues = self._check_citation_placeholders(content)
        issues.extend(citation_issues)

        # Check for proper citation format
        citation_format_issues = self._check_citation_format(content)
        issues.extend(citation_format_issues)

        # Generate suggestions for improvement
        suggestions.extend(self._generate_suggestions(content, issues))

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            suggestions=suggestions
        )

    def _check_for_hallucinations(self, content: str) -> List[str]:
        """
        Check for potential hallucinated information
        """
        issues = []

        # Look for absolute claims without sources
        absolute_claims = re.findall(r'(always|never|all|every|none)\s+\w+', content, re.IGNORECASE)
        if len(absolute_claims) > 3:  # More than 3 might indicate potential overstatements
            issues.append("Multiple absolute claims detected - verify these are accurate and properly sourced")

        # Look for "according to" without specific sources
        unsourced_claims = re.findall(r'according to\s+\w+', content, re.IGNORECASE)
        for claim in unsourced_claims:
            if not any(word in claim.lower() for word in ['study', 'research', 'documentation', 'paper', 'source']):
                issues.append(f"Unsourced claim detected: '{claim}' - add proper citation")

        return issues

    def _check_technical_terminology(self, content: str) -> List[str]:
        """
        Check for proper use of technical terminology
        """
        issues = []

        # Check if technical terms are used consistently
        for term in self.technical_terms:
            # Look for variations in capitalization that might be incorrect
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            matches = re.findall(pattern, content, re.IGNORECASE)
            # Check if the term appears in its correct form
            correct_form_matches = re.findall(r'\b' + re.escape(term) + r'\b', content)

            if len(matches) > len(correct_form_matches) and len(matches) > 0:
                issues.append(f"Term '{term}' may have inconsistent capitalization - ensure proper form is used")

        return issues

    def _check_citation_placeholders(self, content: str) -> List[str]:
        """
        Check for citation placeholders that need to be filled
        """
        issues = []

        for pattern_name, pattern in self.patterns.items():
            if pattern_name == "citation_placeholder":
                placeholders = re.findall(pattern, content)
                if placeholders:
                    issues.append(f"Found {len(placeholders)} citation placeholder(s) that need to be replaced with proper citations")

        return issues

    def _check_citation_format(self, content: str) -> List[str]:
        """
        Check for proper citation format
        """
        issues = []

        # Look for potential citations that might not follow IEEE/APA format
        potential_citations = re.findall(r'\([^)]*\d{4}[^)]*\)', content)  # Year in parentheses might indicate citation

        # This is a basic check - in a real system, we'd have more sophisticated citation detection
        for citation in potential_citations:
            if not self._is_properly_formatted_citation(citation):
                issues.append(f"Potential citation '{citation}' may not follow IEEE/APA format")

        return issues

    def _is_properly_formatted_citation(self, citation: str) -> bool:
        """
        Check if a citation follows proper format (simplified check)
        """
        # In a real implementation, this would have more comprehensive validation
        return True  # Placeholder - implement proper validation

    def _generate_suggestions(self, content: str, issues: List[str]) -> List[str]:
        """
        Generate suggestions for improving content quality
        """
        suggestions = []

        if len(content) < 500:
            suggestions.append("Content may be too brief for a comprehensive chapter - consider adding more detail")
        elif len(content) > 15000:  # Too long
            suggestions.append("Content may be too long - consider breaking into multiple sections")

        # Check for balanced content
        word_count = len(content.split())
        if word_count > 1000:
            # Check if content has appropriate headings
            headings = re.findall(r'^#+\s+.*$', content, re.MULTILINE)
            if len(headings) < 3:
                suggestions.append("Long content should have more section headings for better readability")

        return suggestions

# Create a global instance
validation_service = ContentValidationService()