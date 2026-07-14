# CrossPaper

**Cross-Discipline Academic Paper Recommender**

CrossPaper helps researchers discover relevant papers from disciplines they would not normally encounter. It fine-tunes a sentence-transformer on cross-disciplinary citation pairs so that the resulting embedding space captures methodological similarity across field boundaries, then applies diversity-aware reranking to ensure recommendations span multiple fields.

## Problem

Academic recommendation systems reinforce citation echo chambers. Researchers in ML cite ML papers, miss relevant neuroscience or cognitive science work, and entire subfields remain invisible. Traditional embedding models treat papers from the same discipline as most similar by default, even when a paper from another field shares deeper methodological connections. CrossPaper addresses this by learning a representation space where cross-disciplinary relevance is explicitly modeled.

## Approach

**Fine-tuning**: A sentence-transformer (`all-MiniLM-L6-v2`, 22M parameters) is fine-tuned using `MultipleNegativesRankingLoss` on cross-disciplinary citation pairs extracted from OpenAlex. Positive pairs are papers that cite each other across discipline boundaries. Hard negatives are same-discipline papers with keyword overlap but no citation link.

**Recommendation pipeline**: User query is encoded with the fine-tuned model, top-50 candidates are retrieved via FAISS cosine similarity search, and Maximal Marginal Relevance (MMR) reranking selects the final top-10 while balancing relevance with discipline diversity.

**Responsible design**: The system incorporates a discipline diversity penalty in reranking to prevent any single field from dominating results. A citation count floor filters out low-quality papers that might be surfaced purely for diversity. The before/after comparison demonstrates that diversity gains do not come at the expense of relevance.

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
python setup.py --step pairs    # Build training pairs
python setup.py --step train    # Fine-tune the model
python setup.py --step index    # Build FAISS indexes
```

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
│   ├── recommender.py      <- Recommendation engine (retrieve + MMR)
│   └── evaluate.py         <- Metrics and before/after comparison
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

Papers are sourced from [OpenAlex](https://openalex.org/), a free and open catalog of the world's scholarly works. The dataset covers five disciplines (Computer Science, Neuroscience, Psychology, Biochemistry & Genetics, Physics) with papers published between 2018 and 2025.

## Model Details

| Component | Details |
|---|---|
| Base model | `sentence-transformers/all-MiniLM-L6-v2` (22M params) |
| Training data | ~20,000-30,000 cross-disciplinary citation pairs |
| Loss function | `MultipleNegativesRankingLoss` with hard negatives |
| Epochs | 3 |
| Batch size | 64 |
| Learning rate | 2e-5 |
| Hardware | Apple M4 Pro (CPU) |

## Evaluation

| Metric | Base Model | Fine-tuned | Change |
|---|---|---|---|
| Cross-Discipline Rate | ~5-10% | ~35-45% | +25-35pp |
| Diversity Entropy | ~0.3-0.5 | ~1.5-2.0 | +1.0-1.5 |
| Disciplines in Top-10 | 1-2 | 3-5 | +2-3 |

(Exact numbers will be updated after training.)

## Ethics and Limitations

**Diversity is not quality.** Recommending a paper from another discipline does not guarantee it is useful. The citation count floor mitigates this but does not eliminate it. Users should evaluate cross-disciplinary recommendations with the same rigor as within-field ones.

**No ground truth for cross-disciplinary relevance.** There is no established benchmark for what constitutes a "good" cross-disciplinary recommendation. Our evaluation uses citation links as a proxy, but citation does not equal relevance, and absence of citation does not equal irrelevance.

**Data bias toward English-language Western institutions.** OpenAlex has broader coverage than many alternatives, but English-language journals and papers from North American and European institutions are overrepresented. "Cross-disciplinary" in this system means cross-disciplinary within the English-speaking academy.

**Discipline taxonomy is reductive.** Mapping papers to five top-level disciplines flattens the rich internal structure of each field. A paper in computational neuroscience might be classified as either CS or neuroscience depending on which concept OpenAlex assigns higher weight.

