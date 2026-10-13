# KFC Sales Prediction — Decision Tree

A simple Streamlit app for generating predictions with a trained Decision Tree pipeline.

## Files required

Keep these files in the same GitHub repository folder:

- `app.py`
- `requirements.txt`
- `kfc_decision_tree_model.pkl` (exported from your Kaggle notebook)

## 1. Export the model from Kaggle

Run this after training your pipeline:

```python
import joblib
joblib.dump(pipeline, "/kaggle/working/kfc_decision_tree_model.pkl")
```

Download `kfc_decision_tree_model.pkl` from Kaggle's output files and add it to your repository.

## 2. Deploy on Streamlit Community Cloud

1. Push these files to a GitHub repository.
2. Open https://share.streamlit.io/
3. Choose **Create app** and select your repository and branch.
4. Set the main file path to `app.py`.
5. Deploy.

## 3. Use the app

Upload a CSV containing the same feature columns used when training the model.
Do not include the target column (the value the model predicts). Click **Generate predictions**
and download the resulting CSV.

## Important

- The model file must be named exactly `kfc_decision_tree_model.pkl`.
- Only upload trusted model files. Loading pickle/joblib files from untrusted sources can execute code.
- Large model files may exceed GitHub's file-size limits. If that happens, use Git LFS or a model-hosting/storage option.
