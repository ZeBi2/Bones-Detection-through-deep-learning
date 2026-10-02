# 🦴 Bone Fracture Detection — DenseNet121 + Streamlit

A research/educational Streamlit application for binary bone-fracture image classification using the trained **DenseNet121** model.

## Model specification

| Item | Value |
|---|---|
| Architecture | DenseNet121 |
| Input | 224 × 224 × 3 RGB |
| Classes | `fractured`, `not_fractured` |
| Class mapping | `fractured = 0`, `not_fractured = 1` |
| Parameters | 7,300,418 |
| Selection metric | F1-Score |
| Best F1-Score in supplied metadata | 0.8567674113009198 |
| Frontend | Streamlit |

The supplied model metadata states that DenseNet preprocessing is embedded in the saved model, so the app sends raw RGB image pixels to the model after resizing.

## Repository structure

```text
bone-fracture-streamlit/
│
├── app.py
├── best_bones_fracture_model.keras
├── model_config.pkl
├── class_mapping.pkl
├── image_config.pkl
├── best_model_metadata.json
├── class_mapping.json
├── requirements.txt
├── runtime.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
└── README.md
```

> `best_bones_fracture_model.keras` is the trained model and must be copied from the Kaggle working directory into this repository. It was not included in the uploaded source bundle used to build these deployment files.

## Local installation

Use Python 3.11 for the most predictable TensorFlow environment.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Streamlit deployment

Upload/push the complete repository to GitHub and deploy the repository on Streamlit Community Cloud.

The main file is:

```text
app.py
```

The model file must be in the repository root with exactly this name:

```text
best_bones_fracture_model.keras
```

## Application features

- Modern responsive Streamlit interface
- X-ray image upload
- RGB conversion
- 224 × 224 resizing
- DenseNet121 inference
- Fracture / no-fracture prediction
- Confidence display
- Class probability display
- Optional Grad-CAM explanation
- Model metadata display
- Clear research/educational disclaimer

## Important preprocessing note

The supplied deployment metadata states that DenseNet `preprocess_input` is embedded in the saved model. The app therefore does **not** apply DenseNet preprocessing manually before inference.

## Class mapping

```json
{
  "fractured": 0,
  "not_fractured": 1
}
```

## Medical disclaimer

This application is intended for research and educational purposes only. It is not a medical diagnostic device and should not replace assessment by a qualified healthcare professional.
