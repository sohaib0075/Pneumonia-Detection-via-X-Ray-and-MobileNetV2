# Optimized CNN Chest X-ray Pneumonia Classifier

This project focuses on detecting pneumonia in chest X-ray images using deep learning and transfer learning with the MobileNetV2 architecture. Pneumonia is a serious lung infection that often requires early diagnosis, and chest radiography is a standard imaging method used by radiologists. The goal of this project is to automate this classification task by building an AI system capable of distinguishing between NORMAL and PNEUMONIA cases.

A publicly available dataset containing over 5,000 labeled chest X-ray images was used. Images were preprocessed using rescaling, resizing, and data augmentation techniques to improve generalization. The model was built using MobileNetV2 with pretrained ImageNet weights, followed by a custom classification head including GlobalAveragePooling, Dropout for regularization, and a sigmoid output layer.

The model achieved 83% test accuracy with strong precision (91%) on pneumonia cases. Evaluation metrics such as precision, recall, F1-score, and confusion matrix were used. Grad-CAM visualizations were applied to verify that the model focused on the lung regions during prediction, improving interpretability. Early stopping and model checkpointing helped prevent overfitting and ensured optimal performance.
