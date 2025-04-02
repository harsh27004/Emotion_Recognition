Facial emotion recognition (FER) is a crucial aspect of human-computer interaction, enabling systems to interpret and respond to human emotions in real time. This project presents a machine learning-based approach for classifying facial emotions—Happy, Sad, and Surprised—using Mediapipe's FaceMesh for landmark detection and Scikit-learn for classification.
The system consists of two main phases:
1. Data Preprocessing & Training: Facial landmarks are extracted from a dataset of labeled images, normalized, and used to train a supervised machine learning model.
2. Real-Time Emotion Detection: A webcam captures live facial input, extracts facial landmarks, and predicts the emotion using the trained model.
The model achieves a 93% accuracy across all three emotion classes. However, real-time testing highlighted challenges in detecting "Surprised" expressions consistently. To address this, improvements in data normalization, real-time landmark consistency, and lighting conditions are explored. This project demonstrates the potential of AI-driven facial emotion recognition for applications in human-computer interaction, behavioral analysis, and assistive technologies.

Dataset-https://www.kaggle.com/datasets/thienkhonghoc/affectnet
(used only 3 emotions i.e. happy, sad, and surprised)
