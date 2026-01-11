import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
import numpy as np
import sys
import os
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

# Добавляем modAL
sys.path.append(os.path.join(os.getcwd(), 'modAL'))
from modAL.models import ActiveLearner

# 1. Архитектура SpindleNet
class SpindleNet(nn.Module):
    def __init__(self, input_dim=3):
        super(SpindleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

# 2. Загрузка и объединение ВСЕХ данных
data = np.load("full_dataset.npz")
X_all = np.vstack([data['X_train'], data['X_test']]).astype(np.float32)
y_all = np.concatenate([data['y_train'], data['y_test']]).astype(np.int64)

# Нормализация
mean = X_all.mean(axis=0)
std = X_all.std(axis=0)
X_all = (X_all - mean) / std

# Деление 90/10 (как просил пользователь)
X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X_all, y_all, train_size=70686, test_size=7854, stratify=y_all, random_state=42
)

# Начальная выборка для AL (возьмем 100 примеров)
X_train_init, X_pool, y_train_init, y_pool = train_test_split(
    X_train_pool, y_train_pool, train_size=100, stratify=y_train_pool, random_state=42
)

# 3. Настройка нейросети
net = NeuralNetClassifier(
    SpindleNet,
    max_epochs=30,
    lr=0.05,
    batch_size=64,
    device='cpu',
    verbose=0
)

# 4. Инициализация ActiveLearner
learner = ActiveLearner(
    estimator=net,
    X_training=X_train_init, y_training=y_train_init
)

# 5. Усиленный цикл активного обучения (100 итераций по 10 примеров)
n_queries = 100
print(f"Starting Active Learning (90/10 split)...")
print(f"Pool size: {len(X_pool)}, Test size: {len(X_test)}")

for i in range(n_queries):
    query_idx, query_inst = learner.query(X_pool, n_instances=10)
    learner.teach(X_pool[query_idx], y_pool[query_idx])
    
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    
    if (i+1) % 20 == 0:
        y_test_pred = learner.predict(X_test)
        f1 = f1_score(y_test, y_test_pred)
        print(f"Step {i+1}/100: F1-score on Test Set = {f1:.4f}")

# Финальный отчет
y_final_pred = learner.predict(X_test)
print("\n--- FINAL EVALUATION (90/10 RANDOM SPLIT) ---")
print(classification_report(y_test, y_final_pred, target_names=['No Spindle', 'Spindle']))
