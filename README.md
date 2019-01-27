# Credit-Approval-Models
Various machine learning classification models on credit approval

Dataset at: http://archive.ics.uci.edu/ml/datasets/Credit+Approval

## How to Use

### Setup
1. `git clone` project repository
2. Set up a virtual environment using `python3 -m venv /path/to/env`, then `source ./env/bin/activate`
3. `pip install -r requirements.txt` (includes Keras, tensorflow, sklearn)
4. Download dataset into a folder 'dataset/'

### Run
1. Clean dataset using `python3 clean_data.py` (refer to functions for implementation)
2. Run `python3 *MODEL_NAME*.py` for each model's implementation

## Implemention
### Models
1. Decision Tree
2. Multilayer Perceptron
3. Random Forest
4. Convolutional Neural Network

### Techniques
1. Stratified K-fold cross validation split
2. Grid search cross validation
2. Metric evaluation (confusion matrix, F1-scores)
