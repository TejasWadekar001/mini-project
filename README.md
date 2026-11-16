# RobustLeaf-Net: Plant Disease Detection from Leaf Images

Deep-learning plant disease recognition (CNN + transfer learning) designed to narrow the **lab-to-field gap**,
with Grad-CAM explanations, a severity estimate, and an automatic **PDF disease report** served through a web app.

**Authors:** Tejas Wadekar, Vishwesh Kalled, Shrishant Mugalikar, Vedant Balekundri
**Institution:** Anuvartik Mirji Bharatesh Institute of Technology, Belgaum, Karnataka, India

> **Status:** code complete; experimental results are being generated (see [Results](#results)). Run the smoke test first.

## Method
| Component | Idea |
|---|---|
| **SBR** (Saturation-based Background Randomization) | Leaf mask from colour saturation; background replaced by random colour + noise during training, so the model cannot use background shortcuts |
| **MGA** (Mask-Guided Attention) | CBAM spatial attention map is trained to match the automatic leaf mask (no manual labels) |
| **Hierarchical heads** | Joint crop + disease prediction; crop-consistent decoding at inference |
| **Backbone** | MobileNetV2 (or EfficientNet-B0), two-stage fine-tuning |

Loss: `L = L_disease + 0.3 * L_crop + 0.2 * BCE(attention map, leaf mask)`

## Repository structure
```
data.py  models.py  train.py  evaluate.py      training pipeline and models
gradcam.py  severity.py  infer.py  report.py   explanation, severity, PDF report
disease_info.json                              knowledge base (38 PlantVillage classes)
app.py  predict.py                             Streamlit web app / batch field testing
run_ablation.sh  colab_run_all.ipynb           run all experiments (Colab notebook included)
make_paper_figures.py  fill_paper.py           build paper figures and numbers from real runs
make_results_table.py  make_synthetic.py       results table / fake data for a smoke test
paper/main.tex                                 IEEE-format LaTeX paper
docs/                                          research plan, paper template, sample report
Dockerfile  requirements.txt  LICENSE
```

## Installation
Python 3.10-3.12.
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start
**1. Smoke test (no download, about 2 minutes)**
```bash
python make_synthetic.py synthetic_data
python train.py --data_dir synthetic_data --run_name smoke --model transfer --cbam --sbr --attn_sup --hier \
  --no_pretrained --epochs_head 1 --epochs_ft 1 --batch 8 --split_file runs/smoke_splits.json
```

**2. Train on PlantVillage** (Kaggle: `abdallahalidev/plantvillage-dataset`, use the `color` folder; a GPU is strongly recommended, e.g. Google Colab)
```bash
python train.py --data_dir PlantVillage/color --run_name A5_full --model transfer --cbam --sbr --attn_sup --hier
```

**3. Full ablation (A0-A5) with a field-image test set**
```bash
SEED=42 OODMAP=mapping.json bash run_ablation.sh PlantVillage/color path/to/field_images   # also SEED=1, SEED=2
```
| Run | Configuration |
|---|---|
| A0 | CNN from scratch |
| A1 | MobileNetV2 transfer learning |
| A2 | A1 + CBAM |
| A3 | A2 + SBR |
| A4 | A3 + mask-guided attention |
| A5 | A4 + hierarchical heads (full model) |

Or open `colab_run_all.ipynb` in Google Colab to do everything in the cloud.

**4. Use the model**
```bash
streamlit run app.py                                                     # upload photo -> diagnosis, Grad-CAM, PDF report
python predict.py --run_name A5_full --input field_photos/ --out_dir reports/
python evaluate.py --run_name A5_full --data_dir PlantVillage/color --ood_dir field_images --ood_map mapping.json
```
Example report layout: [`docs/Sample_Disease_Report.pdf`](docs/Sample_Disease_Report.pdf) (illustrative values).

## Results
Results will be added after the experiments are run. Generate the table with `python make_results_table.py`.

| Run | In-domain acc (%) | Field acc (%) | Gap (pts) | Size (MB) |
|---|---|---|---|---|
| A0-A5 | pending | pending | pending | pending |

## Deployment
* **Hugging Face Spaces / Streamlit Community Cloud:** put `runs/deploy/best.keras` (your final model), `runs/splits.json` and the code in the repo; pin the TensorFlow version you trained with (and use `tensorflow-cpu` on small free tiers).
* **Docker:** `docker build -t robustleaf .` then `docker run -p 8501:8501 robustleaf`.

## Paper
`paper/main.tex` is an IEEE-format LaTeX paper. After running the experiments:
```bash
python make_paper_figures.py --data_dir PlantVillage/color --field_dir path/to/field_images
python fill_paper.py --field_images N --field_classes M --gpu "GPU name" --device "CPU name"
```
Then compile `paper/` with pdfLaTeX (for example on Overleaf).

## Limitations
The saturation-based leaf mask is a heuristic (it can fail for low-saturation leaves, white backgrounds or several leaves per image); severity is a colour-based estimate; the system assumes a single leaf per image. Treatment advice in the knowledge base is general guidance - follow product labels and local recommendations.

## License
MIT - see [LICENSE](LICENSE).
