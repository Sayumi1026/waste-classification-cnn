# Trained Models

The trained model files are not included in this repository because of their file sizes.

## Models Developed

1. **Custom CNN** – A convolutional neural network developed from scratch.
2. **MobileNetV2** – A transfer learning model using ImageNet-pretrained weights, followed by fine-tuning.

## Model Performance

| Model | Test Accuracy |
|---|---:|
| Custom CNN | 63.95% |
| MobileNetV2 | 81.05% |

## How to Obtain the Models

Run the notebook in Google Colab to train the models and generate the saved model files.

The final MobileNetV2 model is saved as:

`final_mobilenetv2.keras`

The Custom CNN model is saved as:

`custom_cnn.keras`
