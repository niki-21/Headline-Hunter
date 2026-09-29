# Headline Hunter

**Comparing handcrafted NLP features, TF-IDF, and BERT for news classification—with a Flask inference prototype.**

## Project overview

Headline Hunter investigates a practical ML engineering question: **how much predictive performance is gained as a text classifier becomes more computationally expensive?** The project compares two Random Forest pipelines with a fine-tuned BERT model, measuring accuracy, F1, training time, inference time, and serialized model size.

Developed for MSML606 at the University of Maryland, the repository includes experiment code, labeled data, saved Random Forest models, a local Flask interface, and a technical report. Its engineering focus spans feature design, classical machine learning, transformer fine-tuning, benchmarking, and connecting an inference pipeline to a web application.

The current UI serves the **handcrafted Random Forest only**. TF-IDF and BERT are notebook experiments; a model selector is not implemented. The classifier predicts dataset labels from textual patterns—it does not verify claims against external evidence.

## Model approaches and trade-offs

| Approach | Implementation | Strength | Trade-off |
| --- | --- | --- | --- |
| **Handcrafted features + Random Forest (DSA RF)** | Eight features derived using dictionaries, heaps, sets, sorting, and a phrase trie; 200 trees with maximum depth 10. | Compact, inspectable input features and the lowest reported training and prediction times. | Surface-level style and token-overlap signals do not capture nuanced meaning. Its reported serialized forest is larger than the TF-IDF forest. |
| **TF-IDF + Random Forest** | Up to 5,000 vocabulary features from cleaned title/body text, English stop-word removal, and the same forest configuration. | Higher reported accuracy than handcrafted features while retaining short training and prediction times. | A larger input representation with no contextual or word-order modelling. Serving requires the fitted vectorizer as well as the forest. |
| **Fine-tuned BERT** | Hugging Face `bert-base-uncased` with a two-label classification head, padded/truncated tokenization, and two training epochs with batch size eight. | The highest reported accuracy and F1; contextual token representations offer richer language features. | Much longer reported training, slower inference, and a substantially larger model artifact. The fine-tuned checkpoint is not included. |

The reported comparison makes TF-IDF RF a useful middle-ground baseline. BERT trades more compute and storage for higher predictive performance, while handcrafted RF emphasizes explicit feature design and inexpensive prediction. These observations concern this project's experiments, not universal rankings or measured edge-device performance.

### Handcrafted training features

The training notebook uses the following eight features:

- Clickbait-phrase presence detected with a trie.
- Top-five word-length summaries for the title and body, computed with heaps.
- Maximum word-frequency ratios for the title and body, computed with dictionaries.
- Title/body Jaccard similarity, computed with token sets.
- Longest repeated punctuation run in the title, using insertion sort.
- Longest uppercase title word, using merge sort.

Cleaning removes URLs, HTML, bracketed text, punctuation, and excess whitespace, and lowercases text. Features that depend on capitalization or punctuation use the original text.

## Reported results

The original project results below are preserved unchanged. They appear in [the report, Table II](documents/MSML606_Report.pdf) and [the presentation, slide 19](documents/MSML606_Slides.pdf).

| Model | Accuracy | F1 Score | Training Time | Inference Time | Model Size |
| --- | --- | --- | --- | --- | --- |
| DSA RF | 91.02% | 92.11% | 1.94 sec | 0.000008 sec | 10.32 MB |
| TF-IDF | 95.97% | 96.32% | 5.06 sec | 0.000016 sec | 5.01 MB |
| BERT | 97.95% | 98.09% | 3.5 hours | 0.1171 sec/sample | 437.96 MB |

**How to interpret these numbers:** they are reported experiment results, not fresh benchmarks or measured Flask response times. The Random Forest timing code divides batch prediction time by sample count and excludes feature extraction. Its model-size calculation measures the serialized classifier, not a complete application or the TF-IDF vectorizer. The notebook uses binary F1 for positive label `1`.

Retained notebook outputs reflect different runs: for example, TF-IDF records 96.54% accuracy and 96.82% F1, while the BERT evaluation records 0.1232 sec/sample. The report's table remains the project summary; the repository does not establish a single fully reproducible, hardware-normalized comparison.

## Pipeline and application

```text
Labeled article titles + bodies
            |
      Text preparation
            |
   +--------+----------+-------------------+
   |                   |                   |
8 handcrafted      TF-IDF vectors       BERT tokenizer
features               |                   |
   |               Random Forest       Fine-tuned BERT
Random Forest          |                   |
   +-------------------+-------------------+
            |
 Accuracy / F1 / runtime / model-size comparison

Flask prototype: title + body -> app feature extractor
                -> saved DSA forest -> label + confidence
```

The Flask route accepts a title and article body and calls `predict_proba`. It displays **Real**, **Fake**, or **Uncertain** when the largest class probability is below 60%, along with that probability as a percentage. The displayed confidence is a model score, not a calibrated probability that an article is factually correct.

**Current integration caveat:** `app.py` does not reproduce the notebook's training feature schema. It changes feature order and substitutes capitalization ratio, punctuation ratio, and total word count for the trie/punctuation-run/uppercase-word features. Both representations have eight values, so a matching dimensionality does not establish correctness. The UI is a prototype; its predictions should not be treated as reproducing the reported evaluation until feature parity is restored.

## Tech stack

- **ML and NLP:** scikit-learn, Hugging Face Transformers and Datasets, PyTorch.
- **Data processing:** pandas, NumPy, Python dictionaries/sets, `heapq`, regular expressions, and custom trie/sorting implementations.
- **Serving:** Flask, Jinja templates, HTML, CSS, and JavaScript.
- **Experimentation and artifacts:** Jupyter notebooks, joblib serialization, timing and artifact-size measurements.

## Repository structure

```text
.
├── app.py                         # Flask route and inference feature extraction
├── requirements.txt               # Application and model dependencies (unpinned)
├── notebooks/
│   ├── MSML606.ipynb               # Feature engineering, model training, evaluation
│   └── .ipynb_checkpoints/         # Retained notebook checkpoint
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── evaluation.csv
├── model/
│   ├── dsa_rf_model.pkl            # Handcrafted forest loaded by Flask
│   ├── model_temp1.pkl            # Handcrafted forest benchmark artifact
│   └── model_temp.pkl             # TF-IDF forest benchmark artifact
├── templates/index.html           # Input form and prediction display
├── static/style.css               # Additional stylesheet
├── assets/                        # Example UI screenshots
└── documents/
    ├── MSML606_Report.pdf
    └── MSML606_Slides.pdf
```

## Run locally

From the repository root:

```bash
git clone https://github.com/niki-21/Headline-Hunter.git
cd Headline-Hunter
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The app loads the supplied forest, so retraining is not required to start the prototype. Run from the repository root because its model path is relative. Dependency versions are not pinned; loading a saved scikit-learn model may require the version used to create it. Use trusted model files only.

### Explore the experiments

The main notebook is [`notebooks/MSML606.ipynb`](notebooks/MSML606.ipynb). It combines `train.csv` and `test.csv`, removes fully empty cleaned examples, and creates new stratified 80/20 splits. The supplied `evaluation.csv` is not used in the shown training/evaluation flow.

Notebook-only dependencies missing from `requirements.txt` include `datasets` and `psutil`; a Jupyter runtime and the Transformers Trainer's runtime dependencies are also needed. Start the notebook with `notebooks/` as its working directory so `../data/` and `../model/` resolve correctly.

Review cells before running them: they overwrite model artifacts. The final BERT evaluation cell expects `train.csv` in its working directory and a local `bert_output/checkpoint-500`, neither of which matches the preceding cells' paths. The training cell disables checkpoint saving. A complete BERT rerun therefore requires reconciling these paths and saving/providing a trained checkpoint; downloading the pretrained base model alone does not reproduce the reported result.

## Evaluation limits and engineering lessons

- **Preprocessing and splits:** TF-IDF is fitted before its train/test split, allowing held-out text to influence vocabulary and IDF weights. Its split also lacks a fixed seed, unlike the handcrafted split. A stricter comparison would fit preprocessing only on training data and share a held-out split across models.
- **Training/serving consistency:** the Flask feature mismatch is an example of why preprocessing and ordered feature schemas should be packaged with the estimator and verified together.
- **Artifact completeness:** the fitted TF-IDF vectorizer and fine-tuned BERT checkpoint are absent. Classifier files alone are not complete deployment pipelines.
- **Generalization:** topical and source-specific patterns can affect news classifiers. The reported scores do not establish performance on current news, unseen sources, or independently fact-checked claims.
- **Deployment scope:** the repository demonstrates local serving; it does not include production deployment, calibrated confidence, or an implemented multi-model UI.

## Interface examples

![Example Real prediction in the prototype](assets/demo-real.jpg)

![Example Fake prediction in the prototype](assets/demo-fake.jpg)

These screenshots illustrate the interface, not independent evidence of prediction correctness.

## Contributors

Developed by **Anisha Katiyar, Nikita Miller, Aariz Faridi, and Yatish Sikka**. The report attributes the following areas of work:

| Contributor | Focus |
| --- | --- |
| Nikita Miller | TF-IDF vectorization and Random Forest training; Jaccard similarity and stylometric feature design. |
| Yatish Sikka | Data-structure-based feature engineering, preprocessing, and dataset cleaning. |
| Aariz Faridi | BERT tokenization, fine-tuning, evaluation, and consolidated benchmarks. |
| Anisha Katiyar | Literature review, ethical considerations, result visualization, and evaluation support. |

## Dataset and documentation

- [Dataset source: Fake News Classification on Kaggle](https://www.kaggle.com/datasets/aadyasingh55/fake-news-classification).
- [Technical report](documents/MSML606_Report.pdf): methodology, reported benchmarks, trade-offs, and contributions.
- [Project presentation](documents/MSML606_Slides.pdf): experiment summary and interface examples.
