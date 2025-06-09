# Emotion Recognition Project
> Building a Complete Emotion Recognition System through Multi-Model Training, Migration Learning, and Extended Dataset
> 
> 
>  Developed by Ricky Chen - NCHU CS, 2025 Spring Capstone Project
> 

---

## 📁 Project Structure

```
emotion-recognition-project/
│
├── datasets/
│   ├── fer2013/             # 公開資料集
│   ├── raf-db/              # 公開資料集
│   ├── personal_faces/   # 同學貢獻的臉部影像（for transfer learning）
│   └── emotion_extension/   # 擴延學習情緒（如 engagement, confusion, bored）
│
├── models/                  # 存放已訓練模型
│   ├── fer_fundamental/
│   ├── raf-db_fundamental/
│   └── personal_transfer/   #trandger learning
│
├── training/                # 訓練腳本
│   ├── train_base.py
│   ├── train_transfer.py
│   └── train_extension.py
├── requirements.txt
├── README.md
└── .gitignore

```

---

## 📊 Phase 1: Basic Model Training (8 combinations)

|  Model 📁 Dataset |  Dataset |  Val Accuracy |  Training Time |
| --- | --- | --- | --- |
|  ResNet18 |  FER2013 |  0.655 |  739s |
|  VGG16 |  FER2013 |  0.667 |  2463s |
|  EfficientNet-B0 |  FER2013 |  0.662 |  988s |
|  MobileNet-V2 |  FER2013 |  0.627 |  829s |
|  ResNet18 |  RAF-DB |  0.822 |  321s |
|  VGG16 |  RAF-DB |  0.818 |  1067s |
|  EfficientNet-B0 |  RAF-DB |  0.804 |  434s |
|  MobileNet-V2 |  RAF-DB |  0.742 |  365s |

---

## 🔁 Overview of Phase 2 & 3

- **Phase 2**: Transfer Learning using self-taken face data.
- **Phase 3**: Self-constructed extended emotion dataset (e.g. engagement / boredom) and perform migration training.

---

## 📈 Evaluation result: EfficientNet-B0 on RAF-DB

![image.png](attachment:2fbb6505-e159-4078-87cf-e6cd0d783cbc:image.png)

```
Accuracy: 0.8044
F1 Score: 0.8019

Classification Report:
              precision    recall  f1-score   support

       angry       0.76      0.64      0.69       162
     disgust       0.59      0.47      0.52       160
        fear       0.65      0.46      0.54        74
       happy       0.92      0.91      0.92      1185
     neutral       0.72      0.82      0.77       680
         sad       0.77      0.71      0.74       478
    surprise       0.75      0.83      0.79       329

    accuracy                           0.80      3068
   macro avg       0.74      0.69      0.71      3068
weighted avg       0.80      0.80      0.80      3068

```

---

### 🔍 Why choose EfficientNet-B0 on RAF-DB as the base model for migration learning?

 Although **ResNet18 on RAF-DB** achieved the highest Val Accuracy ( **82.2%** ) in the validation set during Phase 1 training, slightly outperforming other models, we finally chose **EfficientNet-B0 on RAF-DB** as the base model for Phase 2 Migratory Learning based on the following considerations:

1. **The F1-score is the most stable**:
    - The evaluation on the full test set shows that the EfficientNet-B0 model achieves **an Accuracy of 80.4%, an F1 Score of 80.2%**, and maintains a good classification ability in most emotion categories, which demonstrates its excellent generalization ability.
    - On the other hand, the ResNet18 model performs well in the validation set, but the evaluation data of the full test set is not yet available.
2. **The model structure is lightweight, which makes training and inference more efficient**:
    - EfficientNet-B0 has fewer parameters and lower computation requirements without sacrificing accuracy, making it suitable for future applications in personal devices or real-time recognition tasks.
3. **Stability and Scalability Considerations**:
    - Consistent performance on different datasets (FER2013 and RAF-DB) makes EfficientNet a more robust base model for transfer learning and extended tasks (e.g., engagement, confusion class recognition).

 Taking the above considerations into account, **EfficientNet-B0 on RAF-DB** is the best choice for both performance and practicability, and it will be the main framework for migratory learning of personal emotion data in the future.

---

## 📦 Dataset Size and Future Recommendations

 During the training and analysis process, I found that **the total sample size of RAF-DB (about 12,000 samples) is significantly smaller than that of FER2013 (about 35,000 samples)**, and the samples are especially sparse in certain emotion categories (e.g., fear and disgust), which may lead to easy overfitting or unstable learning.

 Nevertheless, the face alignment and labeling quality of RAF-DB is generally better than that of FER2013, which enables models such as ResNet18 and EfficientNet-B0 to show stable recognition results on this dataset.

 ✅ **Strategy:**

- Prioritize **happy, sad, and surprise**, which have high sample size and high classification accuracy, for Phase 2 migration learning.
- Utilize **Data Augmentation** to enhance fear/disgust.
- Consider incorporating other open source micro-emotional datasets to supplement as appropriate

---

## 🔁 Phase 2: Transfer Learning

 📹 Use three selfie videos (happy/sad/surprise) to capture face images and organize them into a personal emotion dataset `personal_faces/` 📷 Consider incorporating other open-source micro-emotional datasets as `a` supplement.

### 📷 Face Capture and Categorization Flow

- Use OpenCV to capture video frames and perform face detection.
- Each mood video corresponds to a folder (e.g. personal_faces/happy/)
- Capture one face per n frames and crop and zoom (224x224)

### 🔄 Data Augmentation

- Random rotation, horizontal flip, and brightness change are applied to each face image to enhance generalization capability
- Augmented images exist directly `personal_faces_augmented/`

### 🧠 Migration Learning Training

- Choose EfficientNet-B0 for the base model (from Phase 1, RAF-DB performance is stable).
- Use frozen conv layers + fine-tune classifier head to train personal data.
- Training process saves new model as `models/efficientnet/personal_transfer.pth`

### 🧪 Testing and analyzing results

- Record three new videos (happy/sad/surprise) and capture faces to create a validation set.
- Model inference results:
    - ✅ Accuracy: 59.78
    - ✅ F1 Score: 0.5671

|  Emotion |  Precision |  Recall |  F1-score |  Support |
| --- | --- | --- | --- | --- |
|  happy |  0.61 |  0.90 |  0.73 |  0.73 |
|  happy 0.61 0.90 132 0.46 |  1.00 |  0.30 |  0.46 |  132 |
|  surprise (i.e., not a surprise) |  0.42 |  0.83 |  0.56 |  52 |

![image.png](attachment:d55bf770-0d7c-4046-979a-adff663a356b:image.png)

 📌 Preliminary observations show that the model is good at recognizing happy, but there is a clear confusion between sad/surprise (the confusion matrix shows that sad is often predicted to be surprise), which may be due to blurred expressions or the quality of filming.

### 🔧 Directions for future improvement

1. **Increase data diversity**: especially for the sad category, include samples with different emotion intensity or angle.
2. **Introduce unfamiliar faces for testing**: to verify the generalization ability of the model to other users.
3. **Apply more advanced migration strategies**: e.g. fine-tune more layers or use teacher-student distillation.
4. **Experiment with more video processing techniques**: e.g., lighting equalization, dynamic expression sequence modeling (LSTM, 3D-CNN).

---

## 🔮 Phase 3 Overview: Extended Emotion Class Learning

### ✅ Objectives

- Extend the model to categories other than the basic seven emotions, such as boredom, engagement, and confusion.
- Explore the transfer learning method to migrate the EfficientNet model trained in the previous phase to new categories.

### 📹 Data Collection and Processing

- Three videos (each corresponding to boredom / engagement / confusion) were actually recorded as data sources.
- Face images were extracted using video capture scripts, with each video producing dozens of images.
- Data Augmentation: Includes random flip, brightness and rotation adjustments to increase sample size and generalization.
- Results are stored in the `emotion_extension_augmented/` folder.

### 🧠 Model Design and Training

- Use the `EfficientNet-B0 on RAF-DB` model trained in Phase 1 as the base.
- Change the classification layer output from 7 to 3 categories (boredom, engagement, confusion).
- Freeze the previous feature extractor and only fine-tune the classification layer.
- Automatically slice the complete enhanced dataset with `80% training / 20% validation`.
- Use Adam optimizer and CrossEntropyLoss to train 5 epochs.
- Model saved as `emotion_extension_transfer_split.pth`.

### 📋 Classification Report (Validation Classification Report)

```
              precision    recall  f1-score   support

    boredom       0.80      0.84      0.82        80
   confusion      0.76      0.79      0.77        75
  engagement      0.76      0.69      0.72        81

    accuracy                          0.77       236
   macro avg      0.77      0.77      0.77       236
weighted avg      0.77      0.77      0.77       236
```

### 📊 Confusion Matrix (Confusion Matrix)

![image.png](attachment:da9bfe59-9220-4821-b749-273447d8ae4b:image.png)

- Classification performance was generally stable, with the best prediction for the boredom category.
- There is still some confusion between engagement and confusion, indicating that there is still room for optimization of the model to distinguish between these two types of emotions.

---

## 🔚 Conclusion

This project successfully demonstrates an end-to-end pipeline for facial emotion recognition, combining multiple training stages from basic supervised learning to real-world personalized and expanded emotion modeling. Through strategic use of transfer learning and lightweight CNN backbones, the system achieves competitive accuracy even under limited data. The methodology lays a solid foundation for future research into multi-modal or real-time emotional feedback applications.

## 👇 Dependencies

```
torch
torchvision
matplotlib
scikit-learn
opencv-python
seaborn

```

---

## 👥 Contributions

- Ricky Chen (NCHU CS): Model development, data collection, training & evaluation
- Peers from KSHS: Participated in emotion video recordings

---
