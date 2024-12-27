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

# Διαδρομές φακέλων εικόνων T1 και T2
t1_path = r"C:\Users\zenia\OneDrive\Υπολογιστής\8ο_9o εξ\Συστηματα Υπ. Αποφ. Εργασια\T1 time point"
t2_path = r"C:\Users\zenia\OneDrive\Υπολογιστής\8ο_9o εξ\Συστηματα Υπ. Αποφ. Εργασια\T2 time point"

# Φόρτωση του αρχείου Excel
annotations_path = r"C:\Users\zenia\OneDrive\Υπολογιστής\8ο_9o εξ\Συστηματα Υπ. Αποφ. Εργασια\BREAST ANNOTATIONS.xls"
annotations = pd.read_excel(annotations_path, header=None)

# Καθαρισμός και ανάγνωση ID και ετικετών
annotations_cleaned = annotations.dropna()
ids = annotations_cleaned.iloc[:, 0].astype(int).values  # Πρώτη στήλη: ID
labels = annotations_cleaned.iloc[:, 1].values          # Δεύτερη στήλη: Ετικέτες (π.χ. "Cancer", "Benign")

# Συνδυασμός ID με εικόνες από τους φακέλους
X = []  # Χαρακτηριστικά
y = []  # Ετικέτες

# Αποθηκεύει το μέγεθος της πρώτης εικόνας για κανονικοποίηση
first_image_shape = None

def perform_segmentation(image):
    """
    Εκτελεί αυτόματο segmentation χρησιμοποιώντας thresholding και morphological operations.
    """
    # Μετατροπή σε grayscale (αν δεν είναι ήδη)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Κανονικοποίηση της εικόνας
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Thresholding
    _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations για καθαρισμό
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    return cleaned_mask

# Flag to print one segmented image
printed_segmented_image = False

for id_, label in zip(ids, labels):
    # Δημιουργία διαδρομών εικόνας
    t1_image_path = os.path.join(t1_path, f"ROI_Z1_{id_}.tif")
    t2_image_path = os.path.join(t2_path, f"ROI_Z2_{id_}.tif")
    
    # Έλεγχος αν οι εικόνες υπάρχουν
    if not os.path.exists(t1_image_path) or not os.path.exists(t2_image_path):
        print(f"Δεν βρέθηκαν εικόνες για το ID {id_}. Παραλείπεται.")
        continue

    try:
        # Φόρτωση εικόνων
        t1_image = tiff.imread(t1_image_path)
        t2_image = tiff.imread(t2_image_path)

        # Ελέγχουμε αν οι εικόνες είναι έγχρωμες (3 κανάλια)
        if len(t1_image.shape) > 2:
            t1_image = t1_image[:, :, 0]
        if len(t2_image.shape) > 2:
            t2_image = t2_image[:, :, 0]

        # Καθορίζουμε το μέγεθος αναφοράς
        if first_image_shape is None:
            first_image_shape = t1_image.shape

        # Προσαρμογή μεγέθους εικόνων στο ίδιο μέγεθος
        t1_image = cv2.resize(t1_image, (first_image_shape[1], first_image_shape[0]))
        t2_image = cv2.resize(t2_image, (first_image_shape[1], first_image_shape[0]))

        # Εκτέλεση segmentation
        t1_segmented = perform_segmentation(t1_image)
        t2_segmented = perform_segmentation(t2_image)

        # Εφαρμογή μασκών στις εικόνες
        t1_masked = cv2.bitwise_and(t1_image, t1_image, mask=t1_segmented.astype(np.uint8))
        t2_masked = cv2.bitwise_and(t2_image, t2_image, mask=t2_segmented.astype(np.uint8))

        # Συνδυασμός εικόνων σε μονοδιάστατο διάνυσμα
        combined_image = np.concatenate((t1_masked.flatten(), t2_masked.flatten()))

        # Προσθήκη χαρακτηριστικών και ετικέτας
        X.append(combined_image)
        y.append(label)

        # Print one segmented image
        if not printed_segmented_image:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("T1 Segmented Image")
            plt.imshow(t1_segmented, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.title("T2 Segmented Image")
            plt.imshow(t2_segmented, cmap='gray')
            plt.show()
            printed_segmented_image = True

    except Exception as e:
        print(f"Σφάλμα κατά την επεξεργασία εικόνων για το ID {id_}: {e}")

# Μετατροπή σε NumPy arrays
X = np.array(X)
y = np.array(y)

# Έλεγχος του σχήματος των δεδομένων
print(f"Σχήμα χαρακτηριστικών (X): {X.shape}")
print(f"Σχήμα ετικετών (y): {y.shape}")

# Κανονικοποίηση χαρακτηριστικών και εκπαίδευση μοντέλου
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Διαχωρισμός δεδομένων σε εκπαίδευση και δοκιμή
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Λίστα μοντέλων για δοκιμή
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Αξιολόγηση μοντέλων
best_model = None
best_accuracy = 0

for model_name, model in models.items():
    print(f"\n=== {model_name} ===")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_cv_score = cv_scores.mean()
    
    # Εκπαίδευση του μοντέλου
    model.fit(X_train, y_train)
    
    # Πρόβλεψη στο σετ δοκιμής
    y_pred = model.predict(X_test)
    
    # Υπολογισμός ακρίβειας
    accuracy = model.score(X_test, y_test)
    
    # Αναφορά απόδοσης
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
    
    # Διατήρηση του καλύτερου μοντέλου
    if accuracy > best_accuracy:
        best_model = model
        best_accuracy = accuracy

print(f"Το καλύτερο μοντέλο είναι: {type(best_model).__name__} με ακρίβεια: {best_accuracy:.4f}")

# Ορισμός του φακέλου εξόδου για αποθήκευση των γραφημάτων
output_dir = r"C:\Users\zenia\OneDrive\Υπολογιστής\8ο_9o εξ\Συστηματα Υπ. Αποφ. Εργασια\output"
os.makedirs(output_dir, exist_ok=True)

# πλοταρισμα χαρακτηριστικων
# Ιστογράμματα για Κατανομή Ετικετών (y)
plt.figure(figsize=(8, 5))
sns.countplot(x=y, palette="pastel")
plt.title("Κατανομή Ετικετών (Benign vs Cancer)")
plt.xlabel("Κατηγορία")
plt.ylabel("Πλήθος")
plt.xticks(ticks=[0, 1], labels=["Benign", "Cancer"])
label_histogram_path = os.path.join(output_dir, "label_distribution.png")
plt.savefig(label_histogram_path)
plt.show()

# Ιστογράμματα για Κατανομή Χαρακτηριστικών
num_features_to_plot = 5  # Αριθμός χαρακτηριστικών που θα οπτικοποιηθούν
plt.figure(figsize=(16, 8))

for i in range(num_features_to_plot):
    plt.subplot(1, num_features_to_plot, i + 1)
    sns.histplot(X_scaled[:, i], kde=True, bins=30, color="blue")
    plt.title(f"Χαρακτηριστικό {i+1} Κατανομή")
    plt.xlabel("Τιμή")
    plt.ylabel("Συχνότητα")
    plt.tight_layout()

feature_histogram_path = os.path.join(output_dir, "feature_distributions.png")
plt.savefig(feature_histogram_path)
plt.show()
