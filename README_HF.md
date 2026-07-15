---
title: CrossPaper
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: main.py
pinned: false
license: mit
---

# CrossPaper

Cross-field academic paper recommender. Fine-tuned on cross-disciplinary
citation pairs from OpenAlex to surface research from fields you would not
normally encounter.

- **Search**: enter a research interest, get recommendations spanning multiple fields
- **Before / After**: compare the base model against the fine-tuned model on the same query

Model: `all-MiniLM-L6-v2` fine-tuned with MultipleNegativesRankingLoss on
16,262 embedding-mined hard negative triplets.

| Protocol | Base | Fine-tuned |
|---|---|---|
| Standard negatives | 0.9417 | 0.9657 |
| Adversarial negatives | 0.1062 | 0.5484 |

Source: https://github.com/Kattine/crosspaper
