# LDAR-DTM: Language Model-Guided Dual Alignment and Refinement for Dynamic Topic Models

Anonymous submission for dynamic topic modeling research.

## Overview

LDAR-DTM is a variational dynamic topic model for time-stamped document collections. It extends the CFDTM-style dynamic topic modeling backbone with two semantic grounding mechanisms:

1. **LLM-guided topic-word refinement.** During training, a large language model scores candidate topic words using the topic's recent temporal history. The model maps each LLM score to a target cosine similarity and directly refines the metric geometry between topic embeddings and word embeddings.
2. **PLM-based dual alignment.** A frozen pre-trained language model (PLM) guides both document-topic inference and topic evolution:
   - **Document-topic alignment:** Sinkhorn optimal transport aligns VAE-inferred topic proportions with PLM-induced document-topic targets.
   - **Evolution alignment:** CKA aligns latent topic drift with PLM semantic drift across adjacent time slices.

## Features

- **LLM-guided metric-space refinement:** For each topic-time pair, the model sends top candidate words and historical top words to an LLM. The LLM returns novelty score values in [0, 1]. Each score is mapped to a target cosine value, and the model minimizes an MSE loss between topic-word cosine similarity and the LLM target.
- **Vocabulary-constrained LLM usage:** The LLM is instructed to rank only words already provided in the candidate list. It does not introduce new vocabulary terms.
- **Document-topic alignment:** Uses Sinkhorn optimal transport in PLM space to construct semantic target topic proportions for documents.
- **Evolution alignment:** Uses CKA to align latent topic evolution with semantic evolution measured by PLM topic embeddings.
- **Dynamic topic evaluation:** Includes topic coherence/diversity, temporal metrics, clustering, and classification evaluation utilities.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies in `requirements.txt`:

- Python 3.8+
- PyTorch 2.6.0+cu124
- torchvision 0.21.0+cu124
- numpy 1.26.0
- scipy 1.13.0
- pandas 2.3.1
- scikit-learn 1.7.1
- gensim 4.3.3
- tqdm 4.67.1
- sentence-transformers 5.1.1
- topmost 1.0.2
- openai 2.36.0

## LLM API Setup

The current implementation uses KRouter through the OpenAI-compatible client in `model/LLMGuider.py`.

Set one or more API keys before running with LLM refinement enabled:

```python
import os
os.environ["GOOGLE_API_KEYS"] = "your_key_1,your_key_2,your_key_3"
```

## Quick Start

The default script runs on the NYT dataset:

```bash
python main.py
```

This will:

1. Generate PLM document embeddings with `all-mpnet-base-v2` and save `train_doc_emb.npy` and `test_doc_emb.npy` into the dataset directory.
2. Load the dynamic dataset, including BoW vectors, time indices, labels, vocabulary, word embeddings, and document embeddings.
3. Train `LDAR_DTM` with reconstruction, KL, ETC, UWE, document-topic alignment, evolution alignment, and LLM-guided refinement.
4. Periodically refresh PLM topic embeddings for alignment.
5. After the LLM warm-up epoch, periodically query KRouter for topic-word guidance if `lambda_contrastive > 0`.
6. Save final top words to `top_words.txt`.
7. Print dynamic topic quality and downstream evaluation metrics.

## Dataset Configuration

### Available Datasets

The model supports multiple benchmark datasets used in the paper:

- **NYT**: New York Times articles (2012-2022) - Default dataset
- **NeurIPS**: NeurIPS conference publications (1987-2017)
- **ACL**: ACL Anthology articles (1973-2006)
- **UN**: United Nations session transcripts (1970-2015)
- **WHO**: WHO articles on non-pharmacological interventions (Jan-May 2020)

### Custom Dataset

To use your own dataset, prepare it in the TopMost format and place it in the `./datasets/` directory. The dataset should contain:

- train_texts.txt, test_texts.txt: Document texts, one per line.
- train_times.txt, test_times.txt: Timestamps for each document.
- Pre-computed word embeddings (e.g., from GloVe).
- The framework will generate other necessary files like BoW representations and document embeddings.

### Running on Other Datasets (e.g., ACL, NeurIPS, UN, WHO)

The default configuration is set for the NYT dataset, which includes labels for downstream tasks. To run on other datasets that do not have labels, you need to make the following three changes in `main.py`:

1. Change the dataset directory:

```python
# Change this line
dataset_dir = "./datasets/NYT"

# To your target dataset, for example:
dataset_dir = "./datasets/ACL"
```

2. Disable label reading:

When initializing `DynamicDataset`, set `read_labels` to `False`.

```python
# Change this line
dataset = DynamicDataset(dataset_dir, batch_size=200, read_labels=True, device=device)

# To:
dataset = DynamicDataset(dataset_dir, batch_size=200, read_labels=False, device=device)
```

3. Comment out downstream task evaluation:

Since these datasets do not have labels, the clustering and classification evaluations will fail. Comment out these sections inside the `evaluate_model` function.

```python
def evaluate_model(...):
    # ... (Topic Coherence and Diversity parts are fine)

    # # Evaluate clustering -- COMMENT OUT THIS BLOCK
    # print("Evaluating clustering performance...")
    # cluster = _clustering(test_theta, dataset.test_labels)
    # purity = cluster['Purity']
    # nmi = cluster['NMI']
    # print(f"Clustering Purity: {purity:.4f}")
    # print(f"Clustering NMI: {nmi:.4f}")

    # # Evaluate classification -- COMMENT OUT THIS BLOCK
    # print("Evaluating classification performance...")
    # clf = _cls(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
    # acc = clf['acc']
    # f1 = clf['macro-F1']
    # print(f"Classification Accuracy: {acc:.4f}")
    # print(f"Classification F1-Score: {f1:.4f}")

    # ... (The rest of the function is fine)
```

## Training Configuration

All hyperparameters are currently configured directly in `main.py`.

```python
model = LDAR_DTM(
    # --- Core model and dataset parameters ---
    vocab_size=dataset.vocab_size,
    num_times=dataset.num_times,
    num_topics=50,
    train_time_wordfreq=dataset.train_time_wordfreq.to(device),
    word_embeddings=dataset.pretrained_WE,
    en_units=200,
    dropout=0.01,
    beta_temp=0.7,

    # --- core losses ---
    temperature=0.1,
    weight_neg=7e+7,
    weight_pos=1.0,
    weight_UWE=1.0e+3,
    neg_topk=15,

    # --- PLM dual alignment ---
    plm_model_name="all-mpnet-base-v2",
    align_warm_up_epoch=5,
    align_frequency=5,
    weight_loss_align=30.0,
    align_sinkhorn_alpha=0.15,
    plm_top_k=15,
    evo_warm_up_epoch=5,
    weight_loss_evo=40.0,

    # --- LLM-guided refinement ---
    llm_warm_up_epochs=296,
    lambda_contrastive=500.0,
    krouter_model_name="gpt-4.0-mini",
    llm_max_workers=3,
    llm_contrastive_temperature=0.1,
    llm_guidance_refresh_rate=1,
    llm_top_k=30,
    llm_history_length=3,
    llm_max_retries=2,
    llm_retry_delay=5,
    llm_batch_size=1,
    llm_log_path="./llm_guidance_logs/",

    # --- Vocabulary mappings required by LLMGuider ---
    idx_to_word=dataset.idx_to_word,
    word_to_idx=dataset.word_to_idx,
)
```

Trainer configuration:

```python
trainer = DynamicTrainer(
    model,
    dataset,
    epochs=300,
    learning_rate=0.002,
    batch_size=200,
    log_interval=5,
    verbose=True,
    num_top_words=15,
)
```

## Key Hyperparameters

- `num_topics`: number of topics.
- `align_warm_up_epoch`: first epoch to apply document-topic alignment.
- `align_frequency`: how often PLM topic embeddings are refreshed.
- `weight_loss_align`: weight for Sinkhorn document-topic alignment.
- `align_sinkhorn_alpha`: Sinkhorn regularization for document-topic alignment.
- `evo_warm_up_epoch`: first epoch to apply evolution alignment.
- `weight_loss_evo`: weight for CKA evolution alignment.
- `llm_warm_up_epochs`: first epoch after which LLM guidance can be queried.
- `lambda_contrastive`: weight for LLM-guided metric-space refinement. 

## Output Files

After a successful run:

- `top_words.txt`: final top words for each topic at each time slice.
- `./llm_guidance_logs/`: JSONL logs of LLM responses, when LLM logging is enabled and API calls are made.
- Console Output: Detailed metrics are printed to the console at the end of the run, including Topic Quality (TQ), Temporal Topic Quality (TTQ), Dynamic Topic Quality (DTQ), and more.

## GPU Notes

The current `main.py` intentionally requires CUDA:

```python
if not torch.cuda.is_available():
    raise RuntimeError(...)
device = "cuda"
```

To run on CPU, remove or modify that guard and set `device = "cpu"`. 
