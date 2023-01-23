import xgboost as xgb
import pandas as pd
from sklearn.model_selection import GridSearchCV
import bentoml

# Load the train and test data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Split the train data into features and target
X_train = train_data[['Pclass', 'Sex', 'Age']]
y_train = train_data['Survived']

# Split the test data into features and target
X_test = test_data[['Pclass', 'Sex', 'Age']]
y_test = test_data['Survived']

# Create the XGBoost model
model = xgb.XGBClassifier()

best_params = {
    # placeholder
}
best_model = xgb.XGBClassifier(best_params)
best_model.fit(X_train, y_train)

# Save the model with BentoML
bentoml.xgboost.save_model('titanic_ml',best_model)