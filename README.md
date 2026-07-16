# CrossPaper

**Cross-Discipline Academic Paper Recommender**

CrossPaper helps researchers discover relevant papers from disciplines they would not normally search. It fine-tunes a sentence-transformer on cross-disciplinary citation pairs so that papers with similar methods are placed closer together even when they come from different fields. A diversity-aware reranker then prevents recommendations from collapsing into a single discipline.

## Problem

Academic recommendation systems can reinforce citation echo chambers. A researcher working in machine learning is much more likely to retrieve other ML papers than related work from neuroscience or psychology, even when the underlying ideas are similar. 

General-purpose embedding models are not trained specifically for cross-disciplinary retrieval, so papers from the same field often cluster together. CrossPaper explores whether fine-tuning on cross-disciplinary citation pairs can make related work from different fields easier to retrieve without sacrificing relevance.

## Approach

**Fine-tuning**: We fine-tune all-MiniLM-L6-v2 (22M parameters), a sentence-transformer built on a distilled Transformer encoder.

Training pairs come from OpenAlex. Each positive pair is a cross-disciplinary citation, meaning one paper cites another paper from a different field. Hard negatives are same-field papers that look similar but have no citation relationship. Hard negatives are mined from the base model's nearest neighbors, making each training example focus on papers the model already finds difficult to distinguish.Training encourages the model to rank the cited paper ahead of these difficult distractors. The goal is not to predict citations, but to reshape the embedding space so that cross-disciplinary methodological connections become easier to retrieve.

**Recommendation pipeline**: A user query is encoded with the fine-tuned model and used to retrieve the top 50 candidates from a FAISS index. Maximal Marginal Relevance (MMR) then reranks those candidates to balance relevance and diversity before returning the final top 10.

**Responsible recommendation**: CrossPaper uses MMR to avoid recommending papers from only one discipline. A citation-count threshold also filters out papers that are too lightly cited to be reliable recommendations.

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
| Batch size | 32 |
| Hardware | Apple M4 Pro (MPS) |
| Learning rate | 2e-5 |

## Evaluation

### Triplet accuracy by negative-sampling protocol
We first evaluate the embedding model itself, before considering the reranker. Each test contains one genuine cross-disciplinary citation and one same-field paper that should not be preferred. The model succeeds if it ranks the true citation higher.
We report two evaluation protocols because the difficulty of the negatives changes the measured accuracy.


| Protocol | Base | Fine-tuned | Change |
|---|---|---|---|
| **Standard** (random same-field negatives) | 0.9417 | 0.9657 | +0.0240 |
| **Adversarial** (nearest-neighbor negatives) | 0.1062 | 0.5484 | **+0.4422** |

Looking at both protocols tells us where the improvement comes from.

Under the standard protocol, the base model is already very strong, so there is little room for improvement. Under the adversarial protocol, negatives are mined from the base model's own nearest neighbors, making them much harder. Fine-tuning substantially improves performance on these difficult cases while maintaining performance on the easy ones.


### End-to-end recommendation 

MMR is evaluated at the deployed setting (λ = 0.6).

| Metric | Base | Fine-tuned | Change |
|---|---|---|---|
| Cross-field rate | 0.420 | 0.500 | +0.080 |
| Expected cross-field rate | 0.360 | 0.420 | +0.060 |
| Diversity entropy | 1.175 | 1.309 | +0.133 |
| Fields in top-10 | 3.20 | 3.40 | +0.200 |

The average hides substantial variation across queries. The strongest benchmark query (evolutionary algorithm for complex systems) improves from three represented fields to all five, while narrowly focused computer science queries show much smaller gains.

This is expected. Queries that are naturally interdisciplinary benefit much more than narrowly defined within-field topics.

Moving from pure relevance (λ = 1.0) to the deployed setting (λ = 0.6) reduces the average embedding relevance score by only 1.9% while increasing diversity substantially. Here, relevance is measured by cosine similarity in the embedding space. We do not have human relevance judgments, so this should be interpreted as a retrieval metric rather than a direct measure of recommendation quality.

### What each component contributes

The embedding model and the reranker solve different problems.

The fine-tuned embedding improves which papers are retrieved. MMR decides how much diversity to introduce among those retrieved papers.

Sweeping λ shows that the fine-tuned model consistently produces higher diversity than the base model across the entire relevance–diversity trade-off, while MMR controls where the system operates on that curve.

Fine-tuning improves the candidate pool. MMR decides how diverse the final recommendations should be.

## Ethics and Limitations

**Evaluation depends on the negative sampling strategy** The reported triplet improvement depends on how difficult the negatives are. Random negatives produce high accuracy for both models, while nearest-neighbor negatives create a much harder evaluation. Reporting both protocols makes this dependence explicit instead of highlighting only the larger improvement.

**We found a bug by looking at the whole curve instead of one number** We originally believed fine-tuning had almost no effect on diversity. That turned out to be our own bug. We applied the field penalty multiplicatively, which accidentally rewarded repeated fields whenever the MMR score became negative. We only noticed something was wrong after plotting diversity across different λ values—the curve went in the opposite direction from what MMR should produce. Fixing the reranker roughly doubled the measured diversity gain. Looking at one operating point hid the bug. Looking at the whole curve exposed it.

**Training pairs should match what the app is asked at inference** Our first version was trained on abstract-to-abstract citation pairs, but the application never sees abstracts—it sees short user queries. That mismatch turned out to matter. Switching to title-anchored pairs better matched the deployment setting and improved retrieval quality.

**Diversity is not the same as usefulness** Cross-disciplinary recommendations are only candidates. Whether they are actually useful still requires human judgment. We report diversity and retrieval metrics because no ground-truth benchmark exists for cross-disciplinary recommendation quality.

**Dataset bias** OpenAlex primarily indexes English-language scholarly literature, so the recommendations reflect biases already present in that corpus. Cross-disciplinary recommendations outside this literature are underrepresented.

**Simplified field labels** Each paper is assigned to one of five broad disciplines. Many papers naturally span multiple fields, so this simplification is useful for evaluation but does not fully represent modern interdisciplinary research.

**Retrieval is limited to a pre-built index** The application searches a fixed FAISS index of 49,926 papers rather than querying OpenAlex at runtime. Fine-tuning does not add new papers—it reshapes the embedding space so that cross-disciplinary papers already in the index become easier to retrieve.
Coverage and representation are therefore decoupled: expanding coverage only requires rebuilding the index, not retraining the model.

**Our citation filter also introduces bias.** To reduce low-quality results, the corpus keeps only papers with more than two citations. This improves retrieval
quality, but it also favors work that has already accumulated academic attention. As a result, part of the citation bias we hope to mitigate is already present in the corpus before the recommender ever runs.

