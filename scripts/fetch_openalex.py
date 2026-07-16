"""
Fetch academic papers and citation data from the OpenAlex API.

Pulls papers from five target fields using the Topics system (the active
replacement for the deprecated Concepts system). Filters for papers with
abstracts and cross-field citations, and saves raw JSON responses for
downstream processing.

Data source: OpenAlex (https://openalex.org/) - free, no API key required.
API docs: https://docs.openalex.org/

Topic hierarchy: Domain -> Field -> Subfield -> Topic
  - 4 domains, 26 fields, 254 subfields, ~4500 topics
"""

import json
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm


FIELDS = {
    "computer_science": 17,        # Domain: Physical Sciences
    "neuroscience": 28,            # Domain: Life Sciences
    "psychology": 32,              # Domain: Social Sciences
    "biochemistry_genetics": 13,   # Domain: Life Sciences (molecular bio crossover)
    "physics": 31,                 # Domain: Physical Sciences
}

BASE_URL = "https://api.openalex.org/works"
DATA_DIR = Path("data/raw")
PAPERS_PER_FIELD = 10000
PER_PAGE = 200  # OpenAlex max per page

# Replace with your actual email for polite pool (10 req/sec)
POLITE_EMAIL = "YOUR EMAIL"


class OpenAlexFetcher:
    """Fetches and stores academic papers from the OpenAlex API.

    Uses the Topics system (not deprecated Concepts) for field classification.
    Handles pagination, rate limiting, and filtering for papers with abstracts
    and cross-field citations.
    """

    def __init__(self, output_dir=DATA_DIR, email=POLITE_EMAIL):
        """Initialize the fetcher.

        Args:
            output_dir: Directory to save raw JSON data.
            email: Email for OpenAlex polite pool (higher rate limits).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.email = email
        self.session = requests.Session()

    def _build_params(self, field_id, cursor="*"):
        """Build query parameters for the OpenAlex API.

        Uses primary_topic.field.id to filter by the paper's primary field
        classification in the Topics system.

        Args:
            field_id: OpenAlex field ID (integer, e.g. 17 for Computer Science).
            cursor: Pagination cursor (use '*' for the first page).

        Returns:
            Dictionary of query parameters.
        """
        return {
            "filter": (
                f"primary_topic.field.id:{field_id},"
                "has_abstract:true,"
                "publication_year:2018-2025,"
                "cited_by_count:>2"
            ),
            "select": (
                "id,title,publication_year,topics,primary_topic,"
                "abstract_inverted_index,referenced_works,cited_by_count"
            ),
            "per_page": PER_PAGE,
            "cursor": cursor,
            "mailto": self.email,
        }

    def _reconstruct_abstract(self, inverted_index):
        """Reconstruct abstract text from OpenAlex inverted index format.

        OpenAlex stores abstracts as inverted indexes mapping words to
        their position indices. This reconstructs the original text.

        Args:
            inverted_index: Dictionary mapping words to list of positions.

        Returns:
            Reconstructed abstract string, or None if input is invalid.
        """
        if not inverted_index:
            return None

        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))

        word_positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in word_positions)

    def _extract_paper(self, raw_work, source_field_name):
        """Extract relevant fields from a raw OpenAlex work record.

        Extracts topic/field/domain information from the new Topics system
        rather than the deprecated Concepts.

        Args:
            raw_work: Raw JSON object from the OpenAlex API.
            source_field_name: Name of the field this paper was fetched under.

        Returns:
            Dictionary with cleaned paper data, or None if the paper
            does not meet quality criteria (abstract too short).
        """
        abstract = self._reconstruct_abstract(
            raw_work.get("abstract_inverted_index")
        )
        if not abstract or len(abstract.split()) < 50:
            return None

        # Extract primary topic and field info
        primary_topic = raw_work.get("primary_topic", {}) or {}
        primary_field = primary_topic.get("field", {}) or {}
        primary_domain = primary_topic.get("domain", {}) or {}

        # Extract all topic-field associations for cross-field detection
        topic_fields = []
        for topic in raw_work.get("topics", []):
            if not topic:
                continue
            field_info = topic.get("field", {}) or {}
            domain_info = topic.get("domain", {}) or {}
            score = topic.get("score", 0)
            if score > 0.3:
                topic_fields.append({
                    "topic_id": topic.get("id", ""),
                    "topic_name": topic.get("display_name", ""),
                    "field_id": field_info.get("id"),
                    "field_name": field_info.get("display_name", ""),
                    "domain_name": domain_info.get("display_name", ""),
                    "score": score,
                })

        return {
            "id": raw_work.get("id", ""),
            "title": raw_work.get("title", ""),
            "abstract": abstract,
            "year": raw_work.get("publication_year"),
            "primary_field_id": primary_field.get("id"),
            "primary_field_name": primary_field.get("display_name", ""),
            "primary_domain": primary_domain.get("display_name", ""),
            "source_field": source_field_name,
            "topic_fields": topic_fields,
            "referenced_works": raw_work.get("referenced_works", []),
            "cited_by_count": raw_work.get("cited_by_count", 0),
        }

    def fetch_field(self, field_name, field_id, max_papers=PAPERS_PER_FIELD):
        """Fetch papers for a single field with cursor-based pagination.

        Args:
            field_name: Human-readable name for logging and file naming.
            field_id: OpenAlex field ID (integer).
            max_papers: Maximum number of papers to fetch.

        Returns:
            List of cleaned paper dictionaries.
        """
        papers = []
        cursor = "*"
        output_path = self.output_dir / f"{field_name}.json"

        # Resume from cached data if available
        if output_path.exists():
            with open(output_path, "r") as f:
                cached = json.load(f)
            if len(cached) >= max_papers:
                print(f"  {field_name}: loaded {len(cached)} cached papers")
                return cached
            papers = cached
            print(f"  {field_name}: resuming from {len(papers)} cached papers")

        pbar = tqdm(
            total=max_papers,
            initial=len(papers),
            desc=f"  {field_name}",
            unit="papers",
        )

        consecutive_errors = 0

        while len(papers) < max_papers:
            params = self._build_params(field_id, cursor)

            try:
                response = self.session.get(BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                consecutive_errors = 0
            except requests.RequestException as e:
                consecutive_errors += 1
                if consecutive_errors > 5:
                    print(f"\n  Too many consecutive errors, stopping. Last: {e}")
                    break
                wait_time = min(2 ** consecutive_errors, 30)
                print(f"\n  API error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            data = response.json()
            results = data.get("results", [])

            if not results:
                break

            for work in results:
                paper = self._extract_paper(work, field_name)
                if paper:
                    papers.append(paper)
                    pbar.update(1)
                    if len(papers) >= max_papers:
                        break

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

            # Respect rate limits (polite pool allows 10 req/sec)
            time.sleep(0.15)

            # Periodic save every 2000 papers
            if len(papers) % 2000 < PER_PAGE:
                with open(output_path, "w") as f:
                    json.dump(papers, f)

        pbar.close()

        with open(output_path, "w") as f:
            json.dump(papers, f, indent=2)
        print(f"  {field_name}: saved {len(papers)} papers to {output_path}")

        return papers

    def fetch_all(self):
        """Fetch papers from all target fields.

        Returns:
            Dictionary mapping field names to lists of papers.
        """
        print("\nFetching papers from OpenAlex API (Topics system)...")
        print(f"Target: {PAPERS_PER_FIELD} papers per field\n")

        all_papers = {}
        for name, field_id in FIELDS.items():
            papers = self.fetch_field(name, field_id)
            all_papers[name] = papers

        # Save combined metadata summary
        summary = {
            name: {
                "count": len(papers),
                "sample_title": papers[0]["title"] if papers else "",
            }
            for name, papers in all_papers.items()
        }
        summary_path = self.output_dir / "fetch_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        total = sum(len(p) for p in all_papers.values())
        print(f"\nTotal papers fetched: {total}")
        print(f"Summary saved to {summary_path}")

        return all_papers


def main():
    """Run the data fetching pipeline."""
    fetcher = OpenAlexFetcher()
    fetcher.fetch_all()


if __name__ == "__main__":
    main()
