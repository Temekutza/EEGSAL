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

# 1. Архитектура SpindleNet (теперь 4 входных признака)
class SpindleNetV2(nn.Module):
    def __init__(self, input_dim=4):
        super(SpindleNetV2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

# 2. Загрузка новых данных
data = np.load("full_dataset_v2.npz")
X_train_full = data['X_train'].astype(np.float32)
y_train_full = data['y_train'].astype(np.int64)
X_test = data['X_test'].astype(np.float32)
y_test = data['y_test'].astype(np.int64)

# Очистка от бесконечностей (бывает при расчете эксцесса на пустых участках)
X_train_full = np.nan_to_num(X_train_full, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Нормализация
mean = X_train_full.mean(axis=0)
std = X_train_full.std(axis=0)
X_train_full = (X_train_full - mean) / (std + 1e-9)
X_test = (X_test - mean) / (std + 1e-9)

# 3. Настройка нейросети
net = NeuralNetClassifier(
    SpindleNetV2,
    max_epochs=40,
    lr=0.02,
    batch_size=128,
    device='cpu',
    verbose=0
)

# Инициализация ActiveLearner (100 начальных точек)
learner = ActiveLearner(
    estimator=net,
    X_training=X_train_full[:100], y_training=y_train_full[:100]
)

# 4. Цикл активного обучения (150 итераций)
print("Training AI with Kurtosis and Relative RMS...")
n_queries = 150
for i in range(n_queries):
    query_idx, _ = learner.query(X_train_full[100:], n_instances=10)
    learner.teach(X_train_full[query_idx], y_train_full[query_idx])
    if (i+1) % 50 == 0:
        print(f"Progress: {i+1}/{n_queries} queries done.")

# Сохраняем веса и параметры нормализации для генератора
torch.save(learner.estimator.module_.state_dict(), 'spindlenet_v2.pt')
np.savez("norm_params.npz", mean=mean, std=std)

# Тест
y_pred = learner.predict(X_test)
print("\n--- PERFORMANCE WITH ADVANCED FEATURES ---")
print(classification_report(y_test, y_pred, target_names=['No Spindle', 'Spindle']))
