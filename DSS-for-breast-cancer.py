# author : @Zenia Fragaki
# 25/12/2024

import os
import numpy as np
import pandas as pd
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import kurtosis, skew
import time

# paths
t1_path = r"path to t1"
t2_path = r"path to t2"

# excel load
annotations_path = r"path to excels"
annotations = pd.read_excel(annotations_path, header=None)

# filtering and reading Ids and labels
annotations_cleaned = annotations.dropna()
ids = annotations_cleaned.iloc[:, 0].astype(int).values  # ID
labels = annotations_cleaned.iloc[:, 1].values          # labels (e.g. "Cancer", "Benign")

# ids and images
X = []  # features
y = []  # labels

# save size
first_image_shape = None

def perform_segmentation(image):
    """
    Εκτελεί αυτόματο segmentation χρησιμοποιώντας thresholding και morphological operations.
    """
    # grayscale grayscale 
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # normalize 
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Thresholding
    _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations 
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    return cleaned_mask

def extract_first_order_features(image):
    """
    Extracts first-order statistical features from the image.
    """
    mean = np.mean(image)
    std = np.std(image)
    kurt = kurtosis(image, axis=None)
    skewness = skew(image, axis=None)
    return [mean, std, kurt, skewness]

def extract_second_order_features(image):
    """
    Extracts second-order texture features from the image using GLCM.
    """
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return [contrast, dissimilarity, homogeneity, energy, correlation]

# Flags to print one segmented image of each category
printed_t1_image = False
printed_t2_image = False

for id_, label in zip(ids, labels):
    #paths
    t1_image_path = os.path.join(t1_path, f"ROI_Z1_{id_}.tif")
    t2_image_path = os.path.join(t2_path, f"ROI_Z2_{id_}.tif")
    
    # if exists
    if not os.path.exists(t1_image_path) or not os.path.exists(t2_image_path):
        print(f"Δεν βρέθηκαν εικόνες για το ID {id_}. Παραλείπεται.")
        continue

    try:
        # load data 
        t1_image = tiff.imread(t1_image_path)
        t2_image = tiff.imread(t2_image_path)

        # color checking
        if len(t1_image.shape) > 2:
            t1_image = t1_image[:, :, 0]
        if len(t2_image.shape) > 2:
            t2_image = t2_image[:, :, 0]

        # size 
        if first_image_shape is None:
            first_image_shape = t1_image.shape

        # resizing
        t1_image = cv2.resize(t1_image, (first_image_shape[1], first_image_shape[0]))
        t2_image = cv2.resize(t2_image, (first_image_shape[1], first_image_shape[0]))

        #  segmentation
        t1_segmented = perform_segmentation(t1_image)
        t2_segmented = perform_segmentation(t2_image)

        # mask
        t1_masked = cv2.bitwise_and(t1_image, t1_image, mask=t1_segmented.astype(np.uint8))
        t2_masked = cv2.bitwise_and(t2_image, t2_image, mask=t2_segmented.astype(np.uint8))

        # Extract first-order features
        t1_first_order_features = extract_first_order_features(t1_masked)
        t2_first_order_features = extract_first_order_features(t2_masked)

        # Extract second-order features
        t1_second_order_features = extract_second_order_features(t1_masked)
        t2_second_order_features = extract_second_order_features(t2_masked)

        # feaut
        combined_features = t1_first_order_features + t2_first_order_features + t1_second_order_features + t2_second_order_features

        # labels
        X.append(combined_features)
        y.append(label)

        #segmented image for T1
        if not printed_t1_image:
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1)
            plt.title('T1 Original Image')
            plt.imshow(t1_image, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title('T1 Mask')
            plt.imshow(t1_segmented, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title('T1 Masked Image')
            plt.imshow(t1_masked, cmap='gray')
            plt.axis('off')

            plt.show()
            printed_t1_image = True

        # segmented image for T2
        if not printed_t2_image:
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1)
            plt.title('T2 Original Image')
            plt.imshow(t2_image, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title('T2 Mask')
            plt.imshow(t2_segmented, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title('T2 Masked Image')
            plt.imshow(t2_masked, cmap='gray')
            plt.axis('off')

            plt.show()
            printed_t2_image = True

    except Exception as e:
        print(f"false id {id_}: {e}")

#  NumPy arrays
X = np.array(X)
y = np.array(y)

# checking feat.
print(f"shape of feat (X): {X.shape}")
print(f"shape of labels (y): {y.shape}")

# Convert features to DataFrame and print
X_df = pd.DataFrame(X, columns=[
    'T1_Mean', 'T1_Std', 'T1_Kurtosis', 'T1_Skewness',
    'T2_Mean', 'T2_Std', 'T2_Kurtosis', 'T2_Skewness',
    'T1_Contrast', 'T1_Dissimilarity', 'T1_Homogeneity', 'T1_Energy', 'T1_Correlation',
    'T2_Contrast', 'T2_Dissimilarity', 'T2_Homogeneity', 'T2_Energy', 'T2_Correlation'
])
print(X_df)

# normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#train test 70-30
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# initialize 
best_model = None
best_accuracy = 0

for model_name, model in models.items():
    print(f"\n=== {model_name} ===")
    
    start_time = time.time()
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_cv_score = cv_scores.mean()
    
    # training
    model.fit(X_train, y_train)
    
    # predictions on test set
    y_pred = model.predict(X_test)
    
    # accuracy computing
    accuracy = model.score(X_test, y_test)
    
    end_time = time.time()
    print(f"Computation time for {model_name}: {end_time - start_time} seconds")
    
    # Prints
    print(f"Cross-Validation Accuracy: {mean_cv_score:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Cancer"], yticklabels=["Benign", "Cancer"])
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    # best accuracy
    if accuracy > best_accuracy:
        best_model = model
        best_accuracy = accuracy

print(f"Best model is: {type(best_model).__name__} with accuracy: {best_accuracy:.4f}")


output_dir = r" your path "
os.makedirs(output_dir, exist_ok=True)


plt.figure(figsize=(8, 5))
sns.countplot(x=y, palette="pastel")
plt.title("labeling classifying "category")
plt.ylabel("sum")
plt.xticks(ticks=[0, 1], labels=["Benign", "Cancer"])
label_histogram_path = os.path.join(output_dir, "label_distribution.png")
plt.savefig(label_histogram_path)
plt.show()

#histograms for feautures
num_features_to_plot = 5  
plt.figure(figsize=(16, 8))

for i in range(num_features_to_plot):
    plt.subplot(1, num_features_to_plot, i + 1)
    sns.histplot(X_scaled[:, i], kde=True, bins=30, color="blue")
    plt.title(f"fearure {i+1} distribution")
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.tight_layout()

feature_histogram_path = os.path.join(output_dir, "feature_distributions.png")
plt.savefig(feature_histogram_path)
plt.show()
