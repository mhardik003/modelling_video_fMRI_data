# **Modeling Brain Activity During Naturalistic Movie Watching**

Linking fMRI  +  Video Embeddings

This project investigates how well high‑level video features can explain—and be reconstructed from—human brain activity while participants watch short films. We:

* **Encode** video embeddings → voxel‑wise fMRI
* **Decode** fMRI → video embeddings
* **Classify** which movie is being watched from fMRI alone

The work builds on SRM‑aligned fMRI from 54 participants who viewed four 8–13 min films ( *Iteration* ,  *Defeat* ,  *Growth* ,  *Lemonade* ) and uses three state‑of‑the‑art video encoders (ViViT, ViT‑MAE, Video‑MAE).

---

## **Repository layout**

```
.
├── raw/                       # All runnable code & notebooks
│   ├── pipeline.ipynb         # End‑to‑end encoding / decoding / classification
│   ├── pipeline_.ipynb        # Voxel‑wise (per‑voxel) encoding variant
│   ├── config.py              # Hyper‑parameters & paths
│   ├── data_loader.py         # fMRI & embedding I/O helpers
│   ├── models.py              # MLP / LSTM model definitions
│   ├── train.py | main.py     # CLI entry‑points for grid / sweep runs
│   └── preprocessing.py       # HRF convolution, z‑scoring, voxel selection
├── stimuli_representations/   # Pre‑computed video embeddings (.npy per movie × encoder)
├── visualizations/            # All result figures used in the report / slides
├── Final_Presentation.pdf
├── Report.pdf
└── README.md                  # ← you are here
```

> **Note** Pre‑aligned **brain representations** are not versioned here for size reasons.

> Download them from [here]() and place the unzipped folder beside **stimuli_representations/**.

---

## **Quick start**

```
# 1. Put brain matrices where config expects them
#    (see the note above)

# 2. Launch Jupyter
jupyter lab raw/pipeline.ipynb
```

The notebook runs end‑to‑end on a single GPU.

For voxel‑wise encoding switch to **pipeline_.ipynb**.

### **Command‑line execution**

```
# Standard encoding experiment (MLP, 1000 voxels)
python raw/train.py --mode encode --voxels 1000 --movie iteration

# Decoding experiment
python raw/train.py --mode decode --encoder videomae

# Four‑way movie classification (intra‑subject)
python raw/main.py --mode classify --split intra
```

All arguments are documented in **config.py**.

---

## **Data**

| **Component**           | **Location**              | **Details**                                     |
| ----------------------------- | ------------------------------- | ----------------------------------------------------- |
| **fMRI (SRM‑aligned)** | [Link]() | 1‑s TR; 54 subjects, 4 movies                        |
| **Video embeddings**    | stimuli_representations/        | ViViT / ViT‑MAE / Video‑MAE; 8 fps → 1 Hz pooled |
| **Visualisations**      | visualizations/                 | Correlation/R² curves, voxel maps, model comparisons |

Dataset courtesy of OpenNeuro **ds004516** (“Individual differences in neural event segmentation of continuous experiences”).

---

## **Key results (reproduced by default notebook)**

| **Task**           | **Best setting**          | **Test metric**                |
| ------------------------ | ------------------------------- | ------------------------------------ |
| **Encoding**       | MLP, 1000 top‑variance voxels  | Pearson*r*≈ 0.30 on*Iteration* |
| **Decoding**       | MLP, 10 k voxels, Video‑MAE   | Pearson*r*≈ 0.05                 |
| **Classification** | Intra‑subject, 5‑layer linear | 99.8 % accuracy                  |
|                          | Inter‑subject                  | 29.9 % (vs. 25 % chance)           |

See **Report.pdf** for the full analysis and **Final_Presentation.pdf** for slide‑level summaries.

---

## **Extending the project**

* **Hyper‑parameter sweeps** – edit **config.py** or pass flags to **train.py**
* **Add a new video encoder** – drop the **.npy** embeddings into **stimuli_representations/** and register it in **data_loader.py**
* **Alternative alignment** – replace SRM matrices in the brain download and adjust **preprocessing.py**

---

## **Citation**

If you use this codebase, please cite:

```
@misc{mittal_bhole_2025_brain_movies,
  title        = {Modeling Brain Activity During Naturalistic Movie Watching Using Video Embeddings and fMRI},
  author       = {Hardik Mittal and Gaurav Bhole},
  year         = {2025},
}
```
