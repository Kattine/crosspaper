"""Explore candidate queries for clearer cross-field demos.

Runs the same query list on base and fine-tuned models, then ranks by
diversity gain and prints titles for manual inspection.
"""

import argparse
from collections import Counter
from pathlib import Path

from recommender import CrossPaperRecommender


DEPLOYED_LAMBDA = 0.6
TOP_N = 10

# Candidate queries grouped by target bridge
CANDIDATE_QUERIES = [
    # CS <-> Neuroscience (1,590 pairs)
    "predictive coding in hierarchical networks",
    "spiking neural network temporal dynamics",
    "hebbian learning and synaptic plasticity",
    "attention mechanism in visual processing",
    "reinforcement learning and dopamine reward prediction",

    # CS <-> Physics (2,968 pairs)
    "phase transitions in large scale systems",
    "network topology and community structure",
    "energy landscape and optimization dynamics",
    "information bottleneck and compression",
    "emergence and self-organization in complex systems",

    # CS <-> Psychology (1,940 pairs)
    "bayesian inference in human cognition",
    "cognitive load and working memory limits",
    "human categorization and concept learning",

    # CS <-> Biochem & Genetics (1,367 pairs)
    "protein structure prediction with deep learning",
    "gene regulatory network inference",

    # Neuroscience <-> Psychology (3,115 pairs)
    "selective attention and cognitive control",

    # Biochem <-> Neuroscience (4,356 pairs, the densest bridge)
    "synaptic protein expression and plasticity",
]


class QueryExplorer:
    """Run query sweeps on both models and rank by diversity gain."""

    def __init__(self, lambda_param=DEPLOYED_LAMBDA, top_n=TOP_N):
        """Initialize sweep settings."""
        self.lambda_param = lambda_param
        self.top_n = top_n
        self.results = []

    def sweep(self, queries):
        """Run all queries on both models and sort by entropy gain."""
        outputs = {}
        for name in ("base", "finetuned"):
            print(f"Loading {name}...")
            rec = CrossPaperRecommender()
            rec.load(index_name=name)
            outputs[name] = [
                rec.recommend(q, top_n=self.top_n, lambda_param=self.lambda_param)
                for q in queries
            ]

        results = []
        for i, query in enumerate(queries):
            base, ft = outputs["base"][i], outputs["finetuned"][i]
            results.append({
                "query": query,
                "base_fields": Counter(r["field"] for r in base["recommendations"]),
                "ft_fields": Counter(r["field"] for r in ft["recommendations"]),
                "base_entropy": base["diversity"]["entropy"],
                "ft_entropy": ft["diversity"]["entropy"],
                "base_n": base["diversity"]["num_fields"],
                "ft_n": ft["diversity"]["num_fields"],
                "entropy_gain": ft["diversity"]["entropy"] - base["diversity"]["entropy"],
                "ft_recs": ft["recommendations"],
                "base_recs": base["recommendations"],
            })

        results.sort(key=lambda r: -r["entropy_gain"])
        self.results = results
        return results

    def print_ranking(self):
        """Print candidates ranked by diversity gain."""
        print("\n" + "=" * 92)
        print(f"  RANKED BY ENTROPY GAIN  (lambda={self.lambda_param}, top-{self.top_n})")
        print("=" * 92)
        print(f"  {'query':<48}{'entropy':>18}{'fields':>12}{'gain':>10}")
        print("  " + "-" * 88)
        for r in self.results:
            entropy = f"{r['base_entropy']:.2f} -> {r['ft_entropy']:.2f}"
            fields = f"{r['base_n']} -> {r['ft_n']}"
            flag = "  <--" if r["entropy_gain"] > 0.4 and r["ft_n"] >= 4 else ""
            print(f"  {r['query'][:46]:<48}{entropy:>18}{fields:>12}"
                  f"{r['entropy_gain']:>+10.2f}{flag}")
        print("=" * 92)
        print("  <-- marks candidates with a large gain AND at least 4 fields")

    def print_details(self, n=3):
        """Print cross-field titles for top candidates."""
        for r in self.results[:n]:
            print("\n" + "=" * 92)
            print(f"  {r['query']}")
            print(f"  entropy {r['base_entropy']:.2f} -> {r['ft_entropy']:.2f}   "
                  f"fields {r['base_n']} -> {r['ft_n']}")
            print("=" * 92)

            for label, recs in (("BASE", r["base_recs"]), ("FINE-TUNED", r["ft_recs"])):
                fields = Counter(x["field"] for x in recs)
                print(f"\n  {label}  {dict(fields)}")
                for x in recs:
                    if x["field"] != "computer_science":
                        print(f"    [{x['field'][:12]:12s}] {x['title'][:72]}")

            print("\n  ^ check every non-CS title above: does the label fit the paper?")


def main():
    """Run query sweep and print ranking/details."""
    parser = argparse.ArgumentParser(
        description="Find the clearest cross-field demo query"
    )
    parser.add_argument("--lambda-param", type=float, default=DEPLOYED_LAMBDA)
    parser.add_argument("--details", type=int, default=3,
                        help="How many top candidates to inspect in detail")
    args = parser.parse_args()

    explorer = QueryExplorer(lambda_param=args.lambda_param)
    explorer.sweep(CANDIDATE_QUERIES)
    explorer.print_ranking()
    explorer.print_details(n=args.details)


if __name__ == "__main__":
    main()
