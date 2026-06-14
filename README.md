# Natural Language Processing

A hands-on collection of NLP experiments built with PyTorch and Hugging Face.
The notebooks move from character-level sequence modeling to sentiment
classification, pretrained transformers, inference pipelines, and
English-to-Spanish translation data preparation.

## Projects

### NLP experiments

[`notebooks/nlp.ipynb`](notebooks/nlp.ipynb) explores:

- character-level text generation using a multi-layer GRU trained on Shakespeare;
- custom BPE tokenization with Hugging Face Tokenizers;
- GPT-2 and BERT tokenization;
- IMDB sentiment classification with GRUs and packed sequences;
- frozen BERT embeddings and BERT-GRU classifiers;
- pretrained sentiment-analysis and natural-language-inference pipelines.

Recorded notebook results:

| Experiment | Best saved validation result |
| --- | ---: |
| Shakespeare character GRU | 55.17% character accuracy |
| Packed-sequence IMDB GRU | 85.44% accuracy |
| Frozen BERT embeddings + bidirectional GRU | 83.92% accuracy |
| Frozen BERT encoder + GRU | 88.50% accuracy |

The Shakespeare model also generates new character-level dialogue from a text
prompt.

### English-Spanish attention translator

[`notebooks/nlp_translator.ipynb`](notebooks/nlp_translator.ipynb) implements an
English-to-Spanish sequence-to-sequence translator using:

- a shared BPE tokenizer with a 10,000-token vocabulary;
- two-layer GRU encoder and decoder networks;
- scaled dot-product attention over encoder outputs;
- source padding masks and teacher-forced target sequences;
- greedy autoregressive decoding for inference.

The `ageron/tatoeba_mt_train` corpus is stored in disk-backed Arrow files to
avoid copying the full dataset into RAM.

Verified split sizes:

| Split | Rows |
| --- | ---: |
| Train | 157,839 |
| Validation | 39,460 |
| Test | 24,514 |

The corrected model was trained on an NVIDIA RTX 4050 Laptop GPU. Validation
peaked at epoch 5, so that checkpoint was selected instead of continuing to
overfit.

| Translator result | Value |
| --- | ---: |
| Best validation token accuracy | 66.72% |
| Best epoch | 5 |

Verified translations:

| English | Generated Spanish |
| --- | --- |
| I love you. | Te quiero. |
| Where is the train station? | ¿Dónde está la estación de tren? |
| This is a beautiful day. | Este es un día hermoso. |
| Can you help me? | ¿Puedes ayudarme? |

The previous inference failure was caused by empty floating-point target tensors.
The repaired decoder uses integer token IDs, starts with `<s>`, stops at `</s>`,
and feeds each predicted token back into the decoder.

## Repository Structure

```text
.
|-- notebooks/
|   |-- datasets/shakespeare/shakespeare.txt
|   |-- nlp.ipynb
|   `-- nlp_translator.ipynb
|-- train_translator.py
|-- requirements.txt
`-- README.md
```

## Setup

Python 3.14 was used for the latest translator run. A CUDA-capable GPU is
recommended for model training, but dataset preparation and inference can run
on CPU.

```powershell
git clone https://github.com/shaheeeeeeeeem/natural-language-processing.git
cd natural-language-processing
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name nlp-project --display-name "Python (NLP Project)"
```

Open the repository in Jupyter or VS Code, select **Python (NLP Project)**, and
run the desired notebook from the first cell.

## Training the Translator

The standalone runner performs resumable mixed-precision GPU training and saves
the latest and best checkpoints after every epoch:

```powershell
python train_translator.py --epochs 10 --batch-size 64
```

If a latest checkpoint exists, the command resumes automatically. Use `--fresh`
to start over.

Checkpoints are excluded from Git because each file is approximately 260 MB,
above GitHub's standard file-size limit. The notebook automatically loads
`checkpoints/nmt_attention_best.pt` when it exists; otherwise, it can train a
new model.

## Datasets and Models

- Shakespeare text from the Hands-On Machine Learning companion dataset
- Stanford IMDB reviews from Hugging Face Datasets
- Tatoeba English-Spanish translation pairs from Hugging Face Datasets
- BERT, DistilBERT, and GPT-2 tokenizers/models from Hugging Face Transformers

Downloads are cached locally and are excluded from Git.

## Notes

- `datasets>=5.0.0` is required for the Python 3.14 translator environment.
- On this Windows/Python 3.14 setup, import Hugging Face Datasets before
  PyTorch to avoid a native PyArrow access violation.
- Restart the kernel after changing packages; an existing kernel keeps old
  modules in memory.
- `nlp.ipynb` is an exploratory learning notebook with saved training outputs.
  Its final custom Hugging Face `Trainer` experiment still contains historical
  dependency-workaround errors, while the standalone pretrained pipelines run
  successfully.
- Training results may vary with hardware, package versions, and random seeds.
