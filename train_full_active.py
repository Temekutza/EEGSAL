import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
import numpy as np
import sys
import os
from sklearn.metrics import f1_score, classification_report

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

# 2. Загрузка полных данных
data = np.load("full_dataset.npz")
X_train_full = data['X_train'].astype(np.float32)
y_train_full = data['y_train'].astype(np.int64)
X_test = data['X_test'].astype(np.float32)
y_test = data['y_test'].astype(np.int64)

# Нормализация (по обучающей выборке)
mean = X_train_full.mean(axis=0)
std = X_train_full.std(axis=0)
X_train_full = (X_train_full - mean) / std
X_test = (X_test - mean) / std

# Начальная выборка (возьмем чуть больше - 50 примеров)
indices = np.arange(len(X_train_full))
np.random.shuffle(indices)
init_idx = indices[:50]
pool_idx = indices[50:]

X_train_init = X_train_full[init_idx]
y_train_init = y_train_full[init_idx]
X_pool = X_train_full[pool_idx]
y_pool = y_train_full[pool_idx]

# 3. Настройка нейросети
net = NeuralNetClassifier(
    SpindleNet,
    max_epochs=30,
    lr=0.05,
    batch_size=64,
    device='cpu', # Используем CPU
    verbose=0
)

# 4. Инициализация ActiveLearner
learner = ActiveLearner(
    estimator=net,
    X_training=X_train_init, y_training=y_train_init
)

# 5. Цикл активного обучения (50 итераций по 10 примеров за раз)
n_queries = 50
print(f"Starting Active Learning on 15 subjects (Pool size: {len(X_pool)})...")

for i in range(n_queries):
    # Запрашиваем 10 наиболее неопределенных примеров
    query_idx, query_inst = learner.query(X_pool, n_instances=10)
    
    # Обучаем на них
    learner.teach(X_pool[query_idx], y_pool[query_idx])
    
    # Удаляем из пула
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    
    if (i+1) % 10 == 0:
        # Промежуточный тест на отложенных 4-х пациентах
        y_test_pred = learner.predict(X_test)
        f1 = f1_score(y_test, y_test_pred)
        print(f"Step {i+1}/50: F1-score on NEW subjects (16-19) = {f1:.4f}")

# Финальный отчет
y_final_pred = learner.predict(X_test)
print("\n--- FINAL EVALUATION ON SUBJECTS 16-19 ---")
print(classification_report(y_test, y_final_pred, target_names=['No Spindle', 'Spindle']))
