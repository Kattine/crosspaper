# CrossPaper

**Cross-Discipline Academic Paper Recommender**

CrossPaper helps researchers discover relevant papers from disciplines they would not normally encounter. It fine-tunes a sentence-transformer on cross-disciplinary citation pairs so that the resulting embedding space captures methodological similarity across field boundaries, then applies diversity-aware reranking to ensure recommendations span multiple fields.

## Problem

Academic recommendation systems reinforce citation echo chambers. Researchers in ML cite ML papers, miss relevant neuroscience or cognitive science work, and entire subfields remain invisible. General-purpose embedding models are not optimized for cross-field retrieval: they often place same-field papers closer together even when a paper from another field shares stronger methodological similarities. CrossPaper fine-tunes the embedding model on cross-field citation pairs so that those relationships are better preserved in the learned space.

## Approach

**Fine-tuning**: A sentence-transformer (all-MiniLM-L6-v2, 22M parameters) is fine-tuned using MultipleNegativesRankingLoss on cross-disciplinary citation pairs extracted from OpenAlex. Positive pairs are papers that cite each other across discipline boundaries. Hard negatives are same-discipline papers with keyword overlap but no citation link.

**Recommendation pipeline**: User query is encoded with the fine-tuned model, top-50 candidates are retrieved via FAISS cosine similarity search, and Maximal Marginal Relevance (MMR) reranking selects the final top-10 while balancing relevance with discipline diversity.

**Responsible design**: The system applies a field diversity penalty during reranking so no single field takes over the results. A citation count floor filters out papers that would otherwise be surfaced purely for diversity. The before/after comparison shows that increased diversity does not degrade performance under our triplet retrieval evaluation. We do not have relevance labels, so this is not a claim about recommendation relevance in general.

## Live Demo

**Web app**: [huggingface.co/spaces/zkmine/crosspaper](https://huggingface.co/spaces/zkmine/crosspaper)

## How to Run Locally

```bash
git clone https://github.com/Kattine/crosspaper.git
cd crosspaper
pip install -r requirements.txt

# Run full pipeline (fetch data, build pairs, fine-tune, build index)
python setup.py

# Launch the web app
python main.py
```

To run individual pipeline steps:

```bash
python setup.py --step fetch    # Fetch papers from OpenAlex
python setup.py --step pairs    # Build cross-field citation pairs
python setup.py --step index    # Build base FAISS index (needed for mining)
python setup.py --step mine     # Mine embedding-based hard negatives
python setup.py --step train    # Fine-tune the model
python setup.py --step index    # Rebuild indexes with the fine-tuned model
```

Note that `build_index.py` runs twice: the base index is required as input to
hard negative mining, and the fine-tuned index is built after training.

## Repository Structure

```
crosspaper/
├── README.md               <- This file
├── requirements.txt        <- Python dependencies
├── setup.py                <- Pipeline orchestrator (data -> train -> index)
├── main.py                 <- Gradio web app entry point
├── .gitignore
├── scripts/
│   ├── fetch_openalex.py   <- Pull papers from OpenAlex API
│   ├── build_training_pairs.py <- Construct cross-discipline training pairs
│   ├── fine_tune.py        <- Fine-tune sentence-transformer
│   ├── build_index.py      <- Build FAISS search indexes
│   ├── mine_hard_negatives.py <- Embedding-based hard negative mining
│   ├── recommender.py      <- Recommendation engine (retrieve + MMR)
│   ├── evaluate.py         <- End-to-end metrics and before/after comparison
│   └── evaluate_triplets.py <- Dual-protocol triplet accuracy evaluation
├── models/
│   ├── base/               <- Pre-fine-tune model checkpoint
│   └── fine_tuned/         <- Fine-tuned model checkpoint
├── data/
│   ├── raw/                <- Raw OpenAlex API responses
│   ├── processed/          <- Training pairs, FAISS indexes, metadata
│   └── outputs/            <- Evaluation results and plots
├── notebooks/
│   ├── 01_eda.ipynb        <- Data exploration (not graded)
│   └── 02_embedding_viz.ipynb <- t-SNE visualization (not graded)
└── assets/
    └── pitch/              <- Pitch materials
```

## Data Source

Papers come from [OpenAlex](https://openalex.org/), a free and open catalog of the world's scholarly works. We sampled 10,000 papers from each of five fields (Computer Science, Neuroscience, Psychology, Biochemistry & Genetics, Physics), published between 2018 and 2025, keeping those with an abstract of at least 50 words and more than two citations. After deduplication the corpus holds 49,926 papers.

## Model Details

| Component | Details |
|---|---|
| Base model | `sentence-transformers/all-MiniLM-L6-v2` (22M params) |
| Training data | 16,262 triplets mined from 18,365 cross-field citation pairs |
| Loss function | `MultipleNegativesRankingLoss` with hard negatives |
| Epochs | 3 |
| Max sequence length | 128 |
| Batch size | 64 |
| Learning rate | 2e-5 |
| Hardware | Apple M4 Pro (CPU) |

## Evaluation

### Triplet accuracy by negative-sampling protocol

We evaluate whether the model ranks a genuine cross-field citation above a
same-field topical lookalike. We measure this under two protocols, because the
choice of negatives strongly influences the measured accuracy:

| Protocol | Base | Fine-tuned | Change |
|---|---|---|---|
| **Standard** (random same-field negatives) | 0.9417 | 0.9657 | +0.0240 |
| **Adversarial** (base model's own nearest neighbors) | 0.1062 | 0.5484 | **+0.4422** |

Chance level is 0.50.

Under the standard protocol, fine-tuning causes no regression on cases the base
model already handled. Under the adversarial protocol, each negative is the
same-field paper the base model itself considers most similar, with no citation
link. The base model scores far below chance here, which is what we expect
under this sampling strategy: the negatives are mined from its own nearest
neighbors. Fine-tuning raises accuracy to 54.8%, so the model no longer
confuses these hard negatives with genuine cross-field citations.

The pattern across both protocols is the point. The base model is already
strong on easy retrieval (94.2%), leaving little room to improve there. The
benefit of fine-tuning shows up almost entirely on adversarially constructed
hard negatives. This suggests the learned representation is better at resolving
difficult semantic ambiguities rather than improving similarity matching
overall.

### End-to-end recommendation metrics (with MMR reranking, lambda=0.6)

| Metric | Base | Fine-tuned | Change |
|---|---|---|---|
| Cross-field rate | 0.440 | 0.460 | +0.020 |
| Expected cross-field rate | 0.400 | 0.420 | +0.020 |
| Diversity entropy | 1.033 | 1.099 | +0.066 |
| Fields in top-10 | 2.80 | 2.80 | 0.00 |

Averaged over five benchmark queries spanning all five fields.

### Component attribution

End-to-end metrics understate the embedding contribution because MMR reranking
substantially influences them. Disabling MMR (lambda=1.0, pure relevance)
isolates the embedding effect: on the query "attention mechanism in visual
processing", cross-field rate rises from 0.20 (base) to 0.30 (fine-tuned) — a
50% relative gain. With MMR enabled, the gap narrows to 0.440 vs 0.460, because
the reranker's field diversity penalty raises both models toward a similar
floor regardless of which embeddings feed it.

The two components have distinct jobs: the fine-tuned embeddings supply
cross-field relevance, and MMR supplies diversity.

## Ethics and Limitations

**The evaluation protocol shapes the headline number.** Our reported
improvement (10.6% to 54.8%) depends heavily on how hard we chose to make the
negatives. Mining them from the base model's own nearest neighbors is expected
to produce a substantially lower baseline; a different mining strategy would
yield a different number from the same model. The low baseline shows the mined
negatives are far more challenging than random ones — it does not by itself
show they are the right negatives, and negatives that are too hard risk
introducing false negatives. The evaluation protocol is not a neutral measuring
instrument but a design choice that strongly influences measured performance,
which is not the same as actual capability. We report both protocols so this is
visible rather than hidden behind the larger number.

**Fine-tuning initially made diversity worse, not better.** The first attempt
trained on full abstract-to-abstract pairs and reduced diversity entropy
(1.033 to 0.901). We identified two contributing factors. First, MMR reranking
masked the embedding contribution: end-to-end metrics could not attribute
effects to individual components, and only an ablation with MMR disabled showed
what the embeddings were doing. Second, training on long abstract pairs
mismatched the short free-text queries the app actually receives, degrading the
base model's existing query understanding. Replacing abstract-to-abstract
training pairs with title-anchored pairs aligned the two distributions and
moved entropy positive (+0.066). Component-level gains do not imply
system-level goals are met.

**Diversity is not quality.** Recommending a paper from another field does not
make it useful. The citation count floor mitigates this but does not eliminate
it. Users should evaluate cross-field recommendations with the same rigor as
within-field ones.

**No ground truth for cross-field relevance.** There is no established benchmark
for what constitutes a good cross-field recommendation. Our evaluation uses
citation links as a proxy, but citation does not equal relevance, and absence of
citation does not equal irrelevance.

**Data bias toward English-language Western institutions.** OpenAlex has broader
coverage than many alternatives, but English-language journals and papers from
North American and European institutions are overrepresented. "Cross-field" in
this system means cross-field within the English-speaking academy.

**Field taxonomy is reductive.** Mapping papers to five top-level fields
flattens the internal structure of each. A computational neuroscience paper may
land in either CS or neuroscience depending on which topic OpenAlex weights
higher.

