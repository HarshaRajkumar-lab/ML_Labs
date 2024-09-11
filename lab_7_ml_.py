import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from scipy.stats import uniform, randint

# Load dataset from the Excel file
file_path = "C:\\Users\\Administrator\\Documents\\ML_bird_project\\bird_species_features (1).xlsx"
df = pd.read_excel(file_path)

# Feature and target columns
features = df.iloc[:, 2:]  # Assuming feature columns start from the 3rd column
target = df['species']  # Assuming the 'species' column is the target

# Splitting dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a RandomizedSearchCV for Perceptron
perceptron = Perceptron()

# Define parameter distribution for Perceptron
param_dist_perceptron = {
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': uniform(0.0001, 0.001),  # Regularization strength
    'max_iter': randint(500, 1000),
    'eta0': uniform(0.01, 0.1)  # Learning rate
}

random_search_perceptron = RandomizedSearchCV(perceptron, param_distributions=param_dist_perceptron, n_iter=10, cv=5, random_state=42, n_jobs=-1)
random_search_perceptron.fit(X_train_scaled, y_train)

# Best Perceptron model
best_perceptron = random_search_perceptron.best_estimator_

# Evaluate the best Perceptron model
y_pred_perceptron = best_perceptron.predict(X_test_scaled)
print("Best Perceptron Model:")
print(random_search_perceptron.best_params_)
print(classification_report(y_test, y_pred_perceptron))

# Define a RandomizedSearchCV for MLPClassifier
mlp = MLPClassifier()

# Define parameter distribution for MLP
param_dist_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 50, 50)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam', 'sgd'],
    'alpha': uniform(0.0001, 0.001),
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': randint(500, 1000)
}

random_search_mlp = RandomizedSearchCV(mlp, param_distributions=param_dist_mlp, n_iter=10, cv=5, random_state=42, n_jobs=-1)
random_search_mlp.fit(X_train_scaled, y_train)

# Best MLP model
best_mlp = random_search_mlp.best_estimator_

# Evaluate the best MLP model
y_pred_mlp = best_mlp.predict(X_test_scaled)
print("\nBest MLP Model:")
print(random_search_mlp.best_params_)
print(classification_report(y_test, y_pred_mlp))
