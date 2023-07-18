# CRC-predicting
This project aims to predict liver metastasis by combining machine learning models and a Chinese BERT model, using both clinical numerical data and Chinese text data. The project includes early and late fusion strategies for prediction.

## Repository Contents

- `ML_Models.py`: Contains the seven machine learning models used for prediction, including SVM, KNN, Decision Tree, Random Forest, XGBoost, Extra Trees, and LightGBM.
- `bert_model.py`: Contains the implementation of the Chinese BERT model.
- `early_fusion.py`: Contains the early fusion strategy implementation, which concatenates the features of machine learning models and the BERT model.
- `late_fusion.py`: Contains the late fusion strategy implementation, which fuses the prediction results of machine learning models and the BERT model.
- `main.py`: The main script used to run the project.
- `Supplymentary Material-3.xlsx`: Contains supplementary material for the project.
- `path.txt`: Contains path information.

## Getting Started

To get started with this project, first clone the repository and install the necessary requirements:

```bash
git clone https://github.com/<your_username>/liver-metastasis-prediction.git
cd liver-metastasis-prediction
pip install -r requirements.txt


python
Copy code




