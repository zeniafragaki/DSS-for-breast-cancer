# DSS for discrimination of benign from malignant cases.

Part of university project for decision supporting systens lesson.

This project aims to develop a Decision Support System (DSS) that assists in the classification of medical imaging data (specifically MRI images) into benign and malignant categories.
The system utilizes machine learning models to analyze image features and accurately predict whether a given case is benign or malignant.


#  Breast Tumors


Breast tumors are classified into benign and malignant types based on their growth characteristics and potential to spread. Benign tumors are 
noncancerous, usually grow slowly, and do not spread to other parts of the body. They can often be removed through surgery, with a low risk of recurrence. On the other hand, malignant tumors are cancerous, grow rapidly, and could invade surrounding tissues and spread (metastasize) to other organs. Malignant breast tumors require more aggressive treatment, such as surgery, chemotherapy, or radiation, and are associated with a higher risk of recurrence and spread. Early detection through screening is crucial for improving outcomes.

# Data Preprocessing


The MRI The images undergo automatic segmentation using thresholding and morphological operations to isolate relevant features. The segmented images are then combined into 1D feature vectors, which are stored alongside their respective tumor labels (e.g., "Benign" or "Cancer").
The features are normalized using a standard scaler to ensure consistency in the data for machine learning models. The dataset is split into training and test sets, allowing for model training and evaluation. Several machine learning models, including Random Forest, SVM, KNN, Logistic Regression, and Decision Trees, are trained and evaluated through cross-validation. Model performance is assessed using accuracy scores, classification reports, and confusion matrices.


# Model Selection

Multiple machine learning models are evaluated, including K-Nearest Neighbors (KNN), Random Forest, Support Vector Machines (SVM), Logistic Regression, and Decision Trees.





# Citation



If you use any part of this project in your work, kindly reference it using the following citation:
Fragaki, Z. (2024). An Evaluation of Machine Learning Classifiers for Discrimination of Benign from Malignant Breast Cancer Cases. GitHub. Available at: https://github.com/zeniafragaki/Breast-cancer-discrimination-benign-malignant
