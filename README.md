# Automated Waste Classification Using CNN and Transfer Learning

## Project Overview

This project develops a deep learning-based image classification system to automatically identify different types of waste materials.

The project uses Convolutional Neural Networks (CNNs) and Transfer Learning to classify waste images into six categories. Two models are developed and compared:

1. Custom CNN trained from scratch.
2. MobileNetV2 using Transfer Learning and Fine-tuning.

## Objectives

- Develop an automated waste classification system using deep learning.
- Explore image preprocessing and data augmentation techniques.
- Build a custom CNN architecture.
- Apply transfer learning using MobileNetV2.
- Evaluate and compare both models using classification metrics.
- Predict waste categories from new images.

## Dataset

**Dataset:** TrashNet

The dataset contains 2,527 images belonging to six waste categories.

| Category | Number of images |
|---|---:|
| Cardboard | 403 |
| Glass | 501 |
| Metal | 410 |
| Paper | 594 |
| Plastic | 482 |
| Trash | 137 |
| **Total** | **2,527** |

Dataset source: https://github.com/garythung/trashnet

The dataset was divided into:

- Training: 70%
- Validation: 15%
- Testing: 15%

Stratified splitting was used to preserve class proportions.

## Technologies Used

- Python
- Google Colab
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow

1. Dataset loading and exploration
2. Exploratory Data Analysis (EDA)
3. Image preprocessing and resizing
4. Training, validation and test splitting
5. Data augmentation
6. Custom CNN development and training
7. MobileNetV2 transfer learning
8. MobileNetV2 fine-tuning
9. Model evaluation and comparison
10. Prediction of new waste images

## Models Developed

### 1. Custom CNN

A convolutional neural network was developed and trained from scratch using the TrashNet dataset.

The architecture includes convolutional layers, max pooling, global average pooling, dense layers and dropout.

### 2. MobileNetV2

MobileNetV2 was used with pretrained ImageNet weights.

Two training stages were performed:

- Feature extraction: pretrained layers were frozen while the new classification layers were trained.
- Fine-tuning: selected pretrained layers were unfrozen and trained with a smaller learning rate.

## Model Performance

The models were evaluated using the same independent test dataset.

| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| Custom CNN | 63.95% | 0.9852 |
| MobileNetV2 | **81.05%** | **0.5539** |

MobileNetV2 outperformed the Custom CNN, achieving an improvement of approximately 17.11 percentage points in test accuracy.

### MobileNetV2 Classification Performance

| Category | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Cardboard | 0.89 | 0.98 | 0.94 |
| Glass | 0.74 | 0.72 | 0.73 |
| Metal | 0.78 | 0.92 | 0.84 |
| Paper | 0.96 | 0.83 | 0.89 |
| Plastic | 0.77 | 0.66 | 0.71 |
| Trash | 0.54 | 0.75 | 0.62 |

## Results and Visualizations

The `results/` directory contains the model comparison chart, confusion matrix and classification reports.

## Repository Structure

```text
waste-classification-cnn/
│
├── notebooks/
│   └── waste_classification.ipynb
│
├── results/
│   ├── model_comparison.png
│   ├── mobilenet_confusion_matrix.png
│   ├── mobilenet_classification_report.csv
│   └── model_comparison.csv
│
├── models/
│   └── README.md
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Limitations

- The dataset contains a relatively small number of images.
- The classes are imbalanced, particularly the trash category.
- Some waste categories have visually similar characteristics.
- Performance on real-world images may differ from performance on the TrashNet dataset.

## Future Improvements

- Increase the number and diversity of training images.
- Explore additional pretrained architectures.
- Improve classification performance for underrepresented categories.
- Develop a user-friendly web application for waste classification.

🚀 Live Application

Try the deployed EcoSort AI waste classification application:

Open EcoSort AI – Waste Classification App
https://waste-classification-cnn-3pege9nnd5ncxodz6yqlvo.streamlit.app/

Upload an image of waste and let the trained MobileNetV2 model predict its category.

## Author

Developed as an academic deep learning project.

## Acknowledgements

Thanks to the creators of the TrashNet dataset for making the image dataset publicly available.

