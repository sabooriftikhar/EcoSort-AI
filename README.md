
# ♻️ EcoSort AI — Deep Learning Waste Classification

EcoSort AI is a deep learning project that classifies waste images into 12 categories using a convolutional neural network (CNN) and transfer learning with MobileNetV2.

The project demonstrates how computer vision can help identify different types of waste and support waste sorting and recycling awareness.

## 🎯 Objectives

- Develop an image classification model for waste recognition.
- Apply transfer learning using MobileNetV2.
- Train and evaluate the model using a labeled image dataset.
- Save the trained model and make predictions on new images.
- Build an interactive web application using Streamlit.
- Visualize training accuracy, loss, and classification results.

## 🗂️ Dataset

**Dataset:** Garbage Classification — 12 Classes

**Source:** [Kaggle Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

The dataset contains images belonging to 12 waste-related categories:

1. Battery
2. Biological
3. Cardboard
4. Clothes
5. Green Glass
6. Brown Glass
7. White Glass
8. Metal
9. Paper
10. Plastic
11. Shoes
12. Trash

The dataset is loaded and processed using TensorFlow's image dataset utilities. Images are resized to 160 × 160 pixels and divided into training and validation subsets.

Please refer to the original Kaggle dataset page for licensing and attribution information.

## 🧠 Model Architecture

EcoSort AI uses MobileNetV2 as its feature extraction backbone.

The classification pipeline includes:

- Image resizing and preprocessing
- Data augmentation
- Pretrained MobileNetV2 feature extractor
- Global average pooling
- Dropout regularization
- Dense classification layer with softmax activation

The pretrained feature extractor is frozen during the initial training phase, while the classification head learns to distinguish the selected waste categories.

### Training Configuration

| Parameter | Value |
|---|---|
| Model | MobileNetV2 |
| Framework | TensorFlow / Keras |
| Input image size | 160 × 160 |
| Batch size | 32 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Loss function | Sparse categorical cross-entropy |
| Maximum epochs | 8 |
| Number of classes | 12 |
| Data split | 80% training / 20% validation |

Early stopping is used to help reduce unnecessary training when validation loss stops improving.

## 📊 Evaluation

The project generates the following evaluation outputs:

- Training and validation accuracy
- Training and validation loss
- Classification report
- Confusion matrix

Actual model results should be recorded after training.

**Training accuracy:** Update after training  
**Validation accuracy:** Update after training  
**Epochs completed:** Update after training

Validation accuracy is not the same as performance on an independent test dataset. A separate test set is recommended for a more reliable estimate of generalization.

## 🚀 Features

- Upload waste images in JPG, JPEG, PNG, or WEBP format
- Predict one of 12 waste categories
- Display the model's confidence score
- Show the top three predicted categories
- View training and validation metrics
- Display accuracy and loss graphs
- Explore the confusion matrix
- Access the project through a Streamlit web interface

## 🛠️ Technologies Used

- Python
- TensorFlow and Keras
- MobileNetV2
- NumPy
- Pillow
- Streamlit
- Google Colab
- Kaggle
- Git and GitHub

## 📁 Project Structure

```text
EcoSort-AI/
├── app.py
├── model.keras
├── class_names.json
├── metrics.json
├── accuracy.png
├── loss.png
├── confusion_matrix.png
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/EcoSort-AI.git
cd EcoSort-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🌐 Live Demo

**Streamlit Application:** Add your deployed application URL here.

## 📦 Model Weights

The trained Keras model is saved as `model.keras`. The repository also contains the class-name mapping and training metrics required by the application.

The model is trained using the dataset referenced above. It should be loaded and tested before deployment to ensure that the model, class mapping, and preprocessing pipeline are consistent.

## 📄 Project Report

Add the link to the final project report PDF here.

## 🎥 Presentation and Video

- **Presentation:** Add your presentation link here.
- **YouTube Video:** Add your public or unlisted video link here.



## ⚠️ Limitations

- Predictions depend on the quality and diversity of the training dataset.
- Images containing multiple types of waste may be difficult to classify correctly.
- Model confidence does not guarantee that a prediction is correct.
- The system is intended for educational demonstration and is not a substitute for professional waste-management assessment.

## 🔮 Future Improvements

- Fine-tune selected MobileNetV2 layers
- Evaluate performance on an independent test dataset
- Expand the dataset with more diverse images
- Add explainable AI visualizations
- Explore object detection for images containing multiple waste items
- Improve deployment efficiency and inference speed

## 👨‍💻 Author

**Abdul Saboor**

Deep Learning Semester Project

## 📜 Disclaimer

This project was developed for educational purposes to demonstrate deep learning, transfer learning, image classification, and web application deployment.

Dataset credit belongs to the original dataset provider. Please check the dataset license before redistribution.
