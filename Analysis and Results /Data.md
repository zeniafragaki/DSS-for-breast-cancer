# Data Source

The dataset used for this project consists of MRI images obtained 
from the MEDISP Laboratory at the University of West Attica. 
Each image was preprocessed, and relevant features were 
extracted to feed into the machine learning models. The images 
were labeled as either benign or malignant based on expert 
annotations.

# Data Processing

The MRI The images undergo automatic segmentation using thresholding and morphological operations to isolate relevant features. The segmented images are then combined into 1D feature vectors, which are stored alongside their respective tumor labels (e.g., "Benign" or "Cancer").
The features are normalized using a standard scaler to ensure consistency in the data for machine learning models. The dataset is split into training and test sets, allowing for model training and evaluation. Several machine learning models, including Random Forest, SVM, KNN, Logistic Regression, and Decision Trees, are trained and evaluated through cross-validation. Model performance is assessed using accuracy scores, classification reports, and confusion matrices.
