# Medical Image Segmentation using U-Net  

---

## 1. Introduction  

### Overview of the Problem  
Medical image segmentation is a critical task in healthcare, enabling the automated identification of anatomical structures and abnormalities in medical imaging modalities like CT scans, MRIs, and X-rays. In this project, the focus is on lung segmentation from chest CT scans.  

Accurate lung segmentation is essential for diagnosing and monitoring lung diseases such as cancer, pneumonia, and chronic obstructive pulmonary disease (COPD). Traditional manual segmentation methods are time-intensive and prone to human error, creating a need for automated, robust solutions.  

This project leverages deep learning, specifically a U-Net model, to perform semantic segmentation. U-Net is a convolutional neural network (CNN) architecture designed for biomedical image segmentation. Its encoder-decoder structure allows the model to learn spatial information and produce pixel-wise segmentation masks, making it ideal for delineating lung regions.  

### Dataset and Objective  
The dataset used is the Kaggle Lung Segmentation Dataset, which consists of grayscale chest CT images and their corresponding binary masks. Each mask indicates whether a given pixel belongs to the lung region (label 1) or the background (label 0).  

**Objective:**  
To build and train a U-Net model that can accurately predict lung segmentation masks for new CT scan images. This automation can serve as a foundation for further medical analysis, including disease classification, 3D modeling of lung structures, or measuring volumetric changes over time.  

---

## 2. Dataset Exploration  

### Understanding the Data  
The dataset consists of grayscale CT images of the chest and their corresponding binary masks.  
- **Input Images:** Grayscale images with pixel values ranging from 0 to 255.  
- **Target Masks:** Binary masks with pixel values of either 0 (background) or 1 (lung region).  

**Dataset Size and Splits:**  
- **Image Dimensions:** Resized to 256×256 pixels for model compatibility.  
- **Training Data:** 80% of the dataset.  
- **Validation Data:** 20% of the dataset.  

### Summary Statistics and Insights  
- **Pixel Value Normalization:** Input images normalized to [0, 1].  
- **Binary Masks:** Retained binary values without normalization.  

**Dataset Size Post-Preprocessing:**  
- Number of Images: `N_total`  
- Training Images: `0.8 × N_total`  
- Validation Images: `0.2 × N_total`  

---

## 3. Pre-processing  

### 3.1 Data Cleaning and Handling Missing Values  
- Verified alignment of CT images with masks.
- Removed any unmatched or missing data points.  

### 3.2 Feature Engineering  
- **Data Augmentation:** Applied transformations like rotations, flips, and zooms to enhance generalization.  

### 3.3 Scaling and Normalization  
- Normalized pixel values to [0, 1] by dividing by 255 for consistent input data.  

---

## 4. Training and Evaluation  

### 4.1 Training the Model  
**Model:** U-Net  
- **Encoder:** Captures spatial information using convolutional and max-pooling layers.  
- **Bottleneck:** Dense feature representation.  
- **Decoder:** Upsamples to original dimensions with skip connections for preserving spatial details.  

### 4.2 Hyperparameters  
- **Learning Rate:** 0.001  
- **Batch Size:** 16  
- **Epochs:** 10  

**Loss Function:** Binary Crossentropy  
**Optimizer:** Adam  

### 4.3 Evaluation Metrics  
- **Accuracy:** Proportion of correctly classified pixels.  
- **Precision:** Ratio of true positives to predicted positives.  
- **Recall:** Ratio of true positives to actual positives.  
- **F1-Score:** Harmonic mean of precision and recall.  

---

## 5. Conclusion  

The U-Net model demonstrated strong performance in lung segmentation from chest CT images.  

**Limitations:**  
- Dataset size and homogeneity limit generalization.  
- Challenges in applying to diverse real-world data.  

**Improvements:**  
- Data augmentation.  
- Use of larger, more diverse datasets.  
- Transfer learning to improve performance with limited data.  

---

## 6. References  

1. Kaggle Lung Segmentation Dataset.  
2. TensorFlow Documentation.  
3. GeeksforGeeks – Image Segmentation Techniques and U-Net Architecture.  
