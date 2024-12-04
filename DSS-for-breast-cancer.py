import os
import numpy as np
import pandas as pd
import tifffile as tiff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Διαδρομή φακέλων T1 και T2
t1_path = r"C:\Users\zenia\OneDrive\Υπολογιστής\8ο_9o εξ\Συστηματα Υπ. Αποφ. Εργασια\T1 time point"
t2_path = r"C:\Users\zenia\OneDrive\Υπολογιστής\8ο_9o εξ\Συστηματα Υπ. Αποφ. Εργασια\T2 time point"

#Loading EXCEL labels for "Cancer" και "Benign"
annotations = pd.read_excel(r"C:\Users\zenia\OneDrive\Υπολογιστής\8ο_9o εξ\Συστηματα Υπ. Αποφ. Εργασια\BREAST ANNOTATIONS.xls", header=None)

#Securing Labelinf(column 1)
labels = annotations.iloc[:, 1].values  # Labeling 2nd Column (Cancer, Benign)

#Lists
X = []
y = []

#Loading and Processing Images
first_image_shape = None  #Sizing Save

for i in range(1, 39):  # Suppose images from 1 to 38
    #paths for T1 and T2
    t1_image_path = os.path.join(t1_path, f"ROI_Z1_{i}.tif")
    t2_image_path = os.path.join(t2_path, f"ROI_Z2_{i}.tif")
    
    # Φορτώνουμε τις εικόνες
    try:
        t1_image = tiff.imread(t1_image_path)  # Διαβάζουμε την εικόνα TIFF
        t2_image = tiff.imread(t2_image_path)  # Διαβάζουμε την εικόνα TIFF
        
        # Ελέγχουμε αν οι εικόνες έχουν περισσότερες από μία διαστάσεις (π.χ., αν είναι έγχρωμες)
        if len(t1_image.shape) > 2:
            t1_image = t1_image[:, :, 0]  # 1d if colorful
        if len(t2_image.shape) > 2:
            t2_image = t2_image[:, :, 0]  # 1d if colorful
        
        # Resizing from 1st image size as default for other
        if first_image_shape is None:
            first_image_shape = t1_image.shape
        
        #Adjusting size for all images 
        t1_image = cv2.resize(t1_image, (first_image_shape[1], first_image_shape[0]))
        t2_image = cv2.resize(t2_image, (first_image_shape[1], first_image_shape[0]))
        
        #Features with concatenate
        combined_image = np.concatenate((t1_image.flatten(), t2_image.flatten()))
        
        # Adding Features in the exact list
        X.append(combined_image)
        y.append(labels[i - 1])  # Labels from EXCEL

    except Exception as e:
        print(f"Erron in loadind image with ID {i}: {e}")
        continue

#Feature size
print(f"Size of image for 1st sample: {len(X[0]) if X else 'N/A'}")
X = np.array(X)

#Labels (Cancer=1, Benign=0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#Train and split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

#Algorithms
models = {
    "KNeighborsClassifier": KNeighborsClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "SVC": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}

#Best accuracy
accuracies = {}
confusion_matrices = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[model_name] = accuracy
    
    # Αποθήκευση της σύγχυσης για πλοτ
    confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)
    
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrices[model_name]}")
    print("-" * 50)

#accuracies
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#confusion matrices
for model_name, cm in confusion_matrices.items():
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

