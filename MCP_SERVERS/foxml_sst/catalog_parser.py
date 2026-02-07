# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Parser for SST_SOLUTIONS.md catalog."""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time


@dataclass
class SSTHelper:
    """Represents an SST helper function."""
    name: str
    category: str
    subcategory: str  # NEW: Subsection within category (e.g., "Target Paths")
    import_path: str
    when_to_use: str
    determinism_impact: str
    common_misuse: str
    example: str = ""
    signature: str = ""
    returns: str = ""

    def matches_query(self, query: str) -> float:
        """
        Score how well this helper matches a search query.
        Returns a score from 0.0 to 1.0.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        score = 0.0

        # Name match (highest weight)
        if query_lower in self.name.lower():
            score += 0.5
        elif any(word in self.name.lower() for word in query_words):
            score += 0.3

        # When to use match (high weight)
        when_lower = self.when_to_use.lower()
        matching_words = sum(1 for word in query_words if word in when_lower)
        score += 0.3 * (matching_words / max(len(query_words), 1))

        # Category/subcategory match
        if query_lower in self.category.lower():
            score += 0.1
        if query_lower in self.subcategory.lower():
            score += 0.15

        # Common misuse match (helps find what NOT to do)
        if query_lower in self.common_misuse.lower():
            score += 0.05

        # Example match
        if query_lower in self.example.lower():
            score += 0.05

        return min(score, 1.0)


class SSTCatalogParser:
    """Parser for SST_SOLUTIONS.md catalog file."""

    def __init__(self, sst_solutions_path: Optional[Path] = None):
        if sst_solutions_path is None:
            # Default path
            sst_solutions_path = Path(__file__).resolve().parents[2] / "INTERNAL" / "docs" / "references" / "SST_SOLUTIONS.md"
        self.path = Path(sst_solutions_path)
        self._catalog: Optional[Dict[str, SSTHelper]] = None
        self._categories: Optional[Dict[str, List[str]]] = None
        self._subcategories: Optional[Dict[str, Dict[str, List[str]]]] = None
        self._last_parse: float = 0
        self._cache_ttl: float = 300.0  # 5 minutes

    def _needs_refresh(self) -> bool:
        """Check if cache needs refresh."""
        if self._catalog is None:
            return True
        return (time.time() - self._last_parse) > self._cache_ttl

    def parse(self) -> Dict[str, SSTHelper]:
        """
        Parse SST_SOLUTIONS.md and return catalog of helpers.

        Handles two formats:
        1. Multi-line format (### `helper_name(...)`)
        2. Inline format (**`helper_name(...)`**)

        Returns:
            Dict mapping helper name to SSTHelper dataclass
        """
        if not self._needs_refresh():
            return self._catalog

        if not self.path.exists():
            self._catalog = {}
            self._categories = {}
            self._subcategories = {}
            return self._catalog

        content = self.path.read_text()

        catalog = {}
        categories = {}
        subcategories = {}

        # Pattern for major category sections (## header)
        category_pattern = re.compile(r'^## ([^#\n]+)', re.MULTILINE)
        # Pattern for subsections (### header)
        subcategory_pattern = re.compile(r'^### ([^`\n]+)$', re.MULTILINE)
        # Pattern for multi-line helper format (### `helper_name(...)`)
        multiline_helper_pattern = re.compile(r'^### `([^`]+)`', re.MULTILINE)
        # Pattern for inline helper format (**`helper_name(...)`**)
        inline_helper_pattern = re.compile(r'^\*\*`([^`]+)`\*\*', re.MULTILINE)

        # Skip categories (non-helper sections)
        skip_categories = {
            "Anti-Patterns", "Adding New Helpers", "Circular Import Prevention",
            "Related References", "Determinism Helpers"
        }

        # Find all category sections
        category_matches = list(category_pattern.finditer(content))

        for i, cat_match in enumerate(category_matches):
            category_name = cat_match.group(1).strip()

            # Skip non-helper categories
            if category_name in skip_categories:
                continue

            # Get content until next category
            start_pos = cat_match.end()
            end_pos = category_matches[i + 1].start() if i + 1 < len(category_matches) else len(content)
            section_content = content[start_pos:end_pos]

            if category_name not in categories:
                categories[category_name] = []
            if category_name not in subcategories:
                subcategories[category_name] = {}

            # Find subsections within this category
            subcat_matches = list(subcategory_pattern.finditer(section_content))

            # Process multi-line format helpers (### `helper_name(...)`)
            multiline_matches = list(multiline_helper_pattern.finditer(section_content))
            for j, helper_match in enumerate(multiline_matches):
                helper = self._parse_multiline_helper(
                    helper_match, multiline_matches, j, section_content,
                    category_name, subcat_matches
                )
                if helper:
                    catalog[helper.name] = helper
                    categories[category_name].append(helper.name)
                    if helper.subcategory:
                        if helper.subcategory not in subcategories[category_name]:
                            subcategories[category_name][helper.subcategory] = []
                        subcategories[category_name][helper.subcategory].append(helper.name)

            # Process inline format helpers (**`helper_name(...)`**)
            inline_matches = list(inline_helper_pattern.finditer(section_content))
            for j, helper_match in enumerate(inline_matches):
                helper = self._parse_inline_helper(
                    helper_match, inline_matches, j, section_content,
                    category_name, subcat_matches
                )
                if helper and helper.name not in catalog:  # Avoid duplicates
                    catalog[helper.name] = helper
                    categories[category_name].append(helper.name)
                    if helper.subcategory:
                        if helper.subcategory not in subcategories[category_name]:
                            subcategories[category_name][helper.subcategory] = []
                        subcategories[category_name][helper.subcategory].append(helper.name)

        self._catalog = catalog
        self._categories = categories
        self._subcategories = subcategories
        self._last_parse = time.time()

        return catalog

    def _find_subcategory(self, position: int, section_content: str, subcat_matches: List) -> str:
        """Find the subcategory for a helper based on its position."""
        current_subcat = ""
        for subcat_match in subcat_matches:
            # Check if this subsection header contains a backtick (helper definition)
            # If so, skip it as it's not a true subcategory
            if '`' in subcat_match.group(0):
                continue
            if subcat_match.start() < position:
                current_subcat = subcat_match.group(1).strip()
            else:
                break
        return current_subcat

    def _parse_multiline_helper(
        self, match, all_matches: List, index: int, section_content: str,
        category: str, subcat_matches: List
    ) -> Optional[SSTHelper]:
        """Parse a multi-line format helper (### `helper_name(...)`)."""
        helper_signature = match.group(1).strip()

        # Extract function name
        name_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', helper_signature)
        if not name_match:
            return None
        helper_name = name_match.group(1)

        # Get content until next helper or end of section
        h_start = match.end()
        h_end = all_matches[index + 1].start() if index + 1 < len(all_matches) else len(section_content)
        helper_content = section_content[h_start:h_end]

        # Find subcategory
        subcategory = self._find_subcategory(match.start(), section_content, subcat_matches)

        # Extract fields (multi-line format uses newlines)
        import_path = self._extract_field(helper_content, r'\*\*Import\*\*:\s*`([^`]+)`')
        when_to_use = self._extract_field(helper_content, r'\*\*When to use\*\*:\s*([^\n]+)')
        determinism_impact = self._extract_field(helper_content, r'\*\*Determinism impact\*\*:\s*([^\n]+)')
        common_misuse = self._extract_field(helper_content, r'\*\*Common misuse\*\*:\s*([^\n]+)')
        example = self._extract_code_example(helper_content)
        returns = self._extract_field(helper_content, r'\*\*Returns\*\*:\s*([^\n]+)')

        return SSTHelper(
            name=helper_name,
            category=category,
            subcategory=subcategory,
            import_path=import_path,
            when_to_use=when_to_use,
            determinism_impact=determinism_impact,
            common_misuse=common_misuse,
            example=example,
            signature=helper_signature,
            returns=returns
        )

    def _parse_inline_helper(
        self, match, all_matches: List, index: int, section_content: str,
        category: str, subcat_matches: List
    ) -> Optional[SSTHelper]:
        """Parse an inline format helper (**`helper_name(...)`**)."""
        helper_signature = match.group(1).strip()

        # Extract function name
        name_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', helper_signature)
        if not name_match:
            return None
        helper_name = name_match.group(1)

        # Get content until next helper or end of section
        h_start = match.end()
        h_end = all_matches[index + 1].start() if index + 1 < len(all_matches) else len(section_content)
        helper_content = section_content[h_start:h_end]

        # Find subcategory
        subcategory = self._find_subcategory(match.start(), section_content, subcat_matches)

        # Extract fields (inline format uses bullet points with dashes)
        import_path = self._extract_bullet_field(helper_content, r'Import')
        when_to_use = self._extract_bullet_field(helper_content, r'When to use')
        determinism_impact = self._extract_bullet_field(helper_content, r'Determinism impact')
        common_misuse = self._extract_bullet_field(helper_content, r'Common misuse')
        example = self._extract_bullet_field(helper_content, r'Example')
        returns = self._extract_bullet_field(helper_content, r'Returns')

        return SSTHelper(
            name=helper_name,
            category=category,
            subcategory=subcategory,
            import_path=import_path,
            when_to_use=when_to_use,
            determinism_impact=determinism_impact,
            common_misuse=common_misuse,
            example=example,
            signature=helper_signature,
            returns=returns
        )

    def _extract_field(self, content: str, pattern: str) -> str:
        """Extract a field value using regex pattern."""
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_bullet_field(self, content: str, field_name: str) -> str:
        """Extract a bullet-point field value (- **Field**: value)."""
        # Pattern for: - **Field**: `value` or - **Field**: value
        pattern = rf'-\s*\*\*{field_name}\*\*:\s*(?:`([^`]+)`|([^\n]+))'
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            # Return backtick value if present, otherwise plain value
            return (match.group(1) or match.group(2) or "").strip()
        return ""

    def _extract_code_example(self, content: str) -> str:
        """Extract code example from content."""
        # Try inline example first: **Example**: `code`
        inline_match = re.search(r'\*\*Example\*\*:\s*`([^`]+)`', content)
        if inline_match:
            return inline_match.group(1).strip()

        # Try code block: ```python ... ```
        block_match = re.search(r'```(?:python)?\s*\n([^`]+)```', content)
        if block_match:
            return block_match.group(1).strip()

        return ""

    def get_categories(self) -> Dict[str, int]:
        """Get all categories with helper counts."""
        self.parse()  # Ensure catalog is loaded
        return {cat: len(helpers) for cat, helpers in (self._categories or {}).items()}

    def get_subcategories(self, category: str) -> Dict[str, int]:
        """Get subcategories within a category with helper counts."""
        self.parse()  # Ensure catalog is loaded
        if not self._subcategories or category not in self._subcategories:
            return {}
        return {subcat: len(helpers) for subcat, helpers in self._subcategories[category].items()}

    def get_helpers_by_category(self, category: str) -> List[SSTHelper]:
        """Get all helpers in a category."""
        self.parse()  # Ensure catalog is loaded

        if not self._categories or category not in self._categories:
            return []

        return [self._catalog[name] for name in self._categories[category] if name in self._catalog]

    def get_helpers_by_subcategory(self, category: str, subcategory: str) -> List[SSTHelper]:
        """Get all helpers in a specific subcategory."""
        self.parse()  # Ensure catalog is loaded

        if not self._subcategories:
            return []
        if category not in self._subcategories:
            return []
        if subcategory not in self._subcategories[category]:
            return []

        return [
            self._catalog[name]
            for name in self._subcategories[category][subcategory]
            if name in self._catalog
        ]

    def search(self, query: str, max_results: int = 10) -> List[SSTHelper]:
        """
        Search helpers by query string.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of matching SSTHelper objects, sorted by relevance
        """
        self.parse()  # Ensure catalog is loaded

        if not self._catalog:
            return []

        # Score all helpers
        scored = []
        for helper in self._catalog.values():
            score = helper.matches_query(query)
            if score > 0:
                scored.append((score, helper))

        # Sort by score (descending) and return top results
        scored.sort(key=lambda x: x[0], reverse=True)
        return [helper for _, helper in scored[:max_results]]

    def get_helper(self, name: str) -> Optional[SSTHelper]:
        """Get a specific helper by name."""
        self.parse()  # Ensure catalog is loaded
        return self._catalog.get(name) if self._catalog else None


# Singleton instance
_parser_instance: Optional[SSTCatalogParser] = None


def get_parser() -> SSTCatalogParser:
    """Get the singleton parser instance."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = SSTCatalogParser()
    return _parser_instance
