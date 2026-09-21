# CONICET Research Semantic Map

This repository contains an end-to-end Natural Language Processing (NLP) and Machine Learning pipeline designed to harvest, analyze, and visualize the scientific landscape of CONICET (Argentina's National Scientific and Technical Research Council).

By extracting metadata from the official repository, generating multilingual semantic embeddings, and applying advanced dimensionality reduction, this project maps researchers into distinct scientific disciplines based purely on the semantic content of their published work.

All the original information has a Creative Commons license, I decided not to upload the paper names, abstracts, filiation or other information to this repository.

## 🗺️ Interactive Visualization

Explore the semantic map of researchers here:

👉 **[View the Interactive HTML Visualization]([https://federicomina-1987.github.io/argentina-research-map/)** 

The visualization is built using Plotly/Bokeh and allows users to explore the 2D semantic space. Each point represents an individual researcher.

* **Color:** Represents the scientific discipline (cluster).
* **Hover Tooltip:** Displays the researcher's name, $x/y$ coordinates, and their top 30 most representative TF-IDF keywords.
* **Navigation:** Fully zoomable and pannable to explore dense sub-disciplines.
```text

conicet-semantic-map/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 00-minado-API.ipynb
│   ├── 01-Analisis-emb.ipynb
│   └── 02-Splitting-and-UMAP.ipynb
│
├── data/ 
│   ├── investigadores_clusters.csv
│   └── investigadores_clusters.json
│
└── docs/
    ├── index.html
```

## 🏗️ Pipeline Architecture

The project is structured into three sequential Jupyter Notebooks, followed by the visualization rendering.

### 1. Data Harvesting (`00-minado-API.ipynb`)

Extracts raw publication metadata directly from the CONICET institutional repository using the Open Archives Initiative Protocol for Metadata Harvesting (OAI-PMH).

* **Protocol:** Queries the `[https://ri.conicet.gov.ar/oai/request](https://ri.conicet.gov.ar/oai/request)` endpoint.
* **Parsing:** Utilizes Python's `xml.etree.ElementTree` with strict OAI and Dublin Core (`oai_dc`) namespace mapping to reliably parse the deeply nested XML responses into structured data.

### 2. Semantic Embeddings Generation (`01-Analisis-emb.ipynb`)

Cleans the harvested text and translates the academic abstracts/titles into high-dimensional mathematical vectors.

* **Model:** Uses `paraphrase-multilingual-mpnet-base-v2` from SentenceTransformers, ensuring accurate semantic mapping across both Spanish and English texts.
* **Hardware Optimization:** Features a custom PyTorch inference loop explicitly optimized for **AMD GPUs**. It bypasses high-level wrappers to manually tokenize text, move tensors to the device, and extract final sentence embeddings in batches, drastically reducing memory bottlenecks and processing time.

### 3. Dimensionality Reduction & Clustering (`02-Splitting and UMAP.ipynb`)

Transforms the high-dimensional embeddings into a structured 2D map and groups researchers by discipline.

* **Keyword Extraction (TF-IDF):** Calculates a Term Frequency-Inverse Document Frequency matrix across combined texts. It filters out common academic terms (`max_df=0.85`) and uses combined English/Spanish stopwords to identify the exact scientific niche of each author.
* **UMAP Projection:** Reduces the 768-dimensional embeddings down to 2D coordinates.
* Configured with `n_neighbors=60` and `min_dist=0.1` to prioritize macroscopic global separation (keeping entire disciplines distinct).
* *Technical Detail:* Injects microscopic Gaussian noise ($\mu=0, \sigma=1e-7$) into the embedding matrix prior to projection to prevent UMAP topological crashes caused by authors with perfectly identical publishing histories.


* **Clustering:** Applies `AgglomerativeClustering` (set to 7 distinct clusters) directly onto the filtered 2D coordinates to strictly define the boundaries of the primary scientific disciplines.

---

## 🚀 Installation & Usage

### Prerequisites

* Python 3.9+
* PyTorch (configured for ROCm if using an AMD GPU)
* `sentence-transformers`, `umap-learn`, `scikit-learn`, `pandas`, `numpy`, `nltk`

### Execution Order

To reproduce the pipeline from scratch, run the notebooks in the following order:

1. `jupyter notebook 00-minado-API.ipynb` (Downloads the dataset to `/conicet_records`)
2. `jupyter notebook 01-Analisis-emb.ipynb` (Generates and saves the 768-D embeddings)
3. `jupyter notebook 02-Splitting and UMAP.ipynb` (Calculates TF-IDF keywords, runs UMAP/Clustering, and exports the final JSON/CSV)
4. *Visualization Script* (Generates the static HTML map from the JSON output)
