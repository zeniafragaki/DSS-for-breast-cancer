# Results 

From these results, we observe that SVM appears to be the most balanced model, offering relatively high accuracy and cross-validation accuracy, indicating both good performance and generalization. Decision Tree, Random Forest and KNN show significant signs of overfitting. Overfitting occurs when a machine learning model performs well on the training data but poorly on unseen data.

![image](https://github.com/user-attachments/assets/dfa29305-1eec-41ac-bf14-6c9b117970d3)


# Confusion Matrices

![image](https://github.com/user-attachments/assets/c7a9e6ec-90cf-4c14-bbbe-c2de988c4345)

Fig.1: Confusion Matrix for SVM


![image](https://github.com/user-attachments/assets/e50b99eb-5094-4fe8-ae7e-8d32d216559f)

Fig.2: Confusion Matrix for Random Forest


![image](https://github.com/user-attachments/assets/81e29deb-59ad-49c1-9581-8322f0bb9170)

Fig.3: Confusion Matric for KNN


![image](https://github.com/user-attachments/assets/16045088-0feb-4269-96f4-37c743263fe9)

Fig.4: Confusion Matrix for Logistic Regression


# Discussion

The results suggest that SVM performs the best in classifying MRI images as benign or malignant in this dataset. The performance of the machine learning models in this study was significantly influenced by the limited dataset size, comprising only 40 MRI images per time point (T1 and T2). This small dataset poses challenges in building robust models capable of generalizing to unseen data. Machine learning models typically require larger datasets to capture meaningful patterns while avoiding overfitting.


# Conclusion


In conclusion, the project demonstrates the potential of machine learning techniques in classifying medical images for benign and malignant detection. The SVM model showed the highest accuracy, but more research is needed to further enhance the system's robustness. Future work will and the integration of more advanced machine learning algorithms.







