# Deployment Checklist

## Before GitHub upload

- [ ] `app.py`
- [ ] `best_bones_fracture_model.keras`
- [ ] `model_config.pkl`
- [ ] `class_mapping.pkl`
- [ ] `image_config.pkl`
- [ ] `best_model_metadata.json`
- [ ] `class_mapping.json`
- [ ] `requirements.txt`
- [ ] `runtime.txt`
- [ ] `.gitignore`
- [ ] `.streamlit/config.toml`
- [ ] `README.md`

## Critical model file

Copy this file from Kaggle:

`/kaggle/working/best_bones_fracture_model.keras`

Place it beside `app.py`.

## Local test

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Git commands

```bash
git init
git add .
git commit -m "Add bone fracture Streamlit deployment"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## Streamlit

Select the GitHub repository and set:

`Main file path: app.py`

Then deploy.
