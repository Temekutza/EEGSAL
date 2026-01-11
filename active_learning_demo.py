import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# Add modAL to path
sys.path.append(os.path.join(os.getcwd(), 'modAL'))

from modAL.models import ActiveLearner

# Load data
df = pd.read_csv("processed_features_0001.csv")
X = df[['rms', 'rel_sigma_power', 'peak_sigma_freq']].values
y = df['target'].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Initial split: small training set, larger pool
X_train_init, X_pool, y_train_init, y_pool = train_test_split(X, y, train_size=10, stratify=y, random_state=42)

# Define Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
gpc = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=10)

# Initialize ActiveLearner
learner = ActiveLearner(
    estimator=gpc,
    X_training=X_train_init, y_training=y_train_init
)

# Active Learning loop
n_queries = 20
history = []

print(f"Initial accuracy: {accuracy_score(y_pool, learner.predict(X_pool)):.4f}")

for i in range(n_queries):
    # Query the most uncertain instance
    query_idx, query_inst = learner.query(X_pool)
    
    # "Label" the instance (get it from our pool labels)
    learner.teach(X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(1,))
    
    # Evaluate
    y_pred = learner.predict(X_pool)
    acc = accuracy_score(y_pool, y_pred)
    f1 = f1_score(y_pool, y_pred)
    history.append((i+1, acc, f1))
    
    # Remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    
    if (i+1) % 5 == 0:
        print(f"Query {i+1}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

# Plot learning curve
history_df = pd.DataFrame(history, columns=['query', 'accuracy', 'f1'])
plt.figure(figsize=(10, 5))
plt.plot(history_df['query'], history_df['f1'], label='F1-score')
plt.plot(history_df['query'], history_df['accuracy'], label='Accuracy')
plt.xlabel('Number of queries')
plt.ylabel('Score')
plt.title('Active Learning Curve (Gaussian Process)')
plt.legend()
plt.grid(True)
plt.savefig('learning_curve.png')
print("Learning curve saved to learning_curve.png")
