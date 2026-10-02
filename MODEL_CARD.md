# Model Card — Bone Fracture Detection

## Model

DenseNet121 binary image classifier.

## Input

RGB image resized to 224 × 224 pixels.

## Classes

- fractured (index 0)
- not_fractured (index 1)

## Supplied model-selection metadata

Selection metric: F1-Score

Best F1-Score: 0.8567674113009198

## Intended use

Research and educational demonstration of image classification and explainability.

## Limitations

This application is not a medical diagnostic device. Performance on a new clinical population, imaging device, hospital, or acquisition protocol is not established by the supplied metadata alone.

## Explainability

The Streamlit frontend provides optional Grad-CAM visualization. The training notebook also contains a broader XAI workflow including Grad-CAM, Grad-CAM++, Score-CAM, Integrated Gradients, LIME, and optional SHAP. The Streamlit app uses Grad-CAM because it is more practical for interactive inference.
