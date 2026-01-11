import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Добавляем modAL
sys.path.append(os.path.join(os.getcwd(), 'modAL'))
from modAL.models import ActiveLearner

# 1. Определение архитектуры нейросети
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

# 2. Загрузка и подготовка данных
df = pd.read_csv("multi_subject_data.csv")
X = df[['rms', 'sigma_power', 'sigma_alpha_ratio']].values.astype(np.float32)
y = df['target'].values.astype(np.int64)

# Нормализация
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Разделение на начальную обучающую выборку и пул
X_train_init, X_pool, y_train_init, y_pool = train_test_split(X, y, train_size=20, stratify=y)

# 3. Обертка нейросети в skorch для совместимости с sklearn/modAL
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = NeuralNetClassifier(
    SpindleNet,
    max_epochs=20,
    lr=0.1,
    batch_size=32,
    device=device,
    verbose=0 # Скрываем логи обучения
)

# 4. Инициализация ActiveLearner
learner = ActiveLearner(
    estimator=net,
    X_training=X_train_init, y_training=y_train_init
)

# 5. Цикл активного обучения
n_queries = 30
print(f"Starting Active Learning with Neural Network on {device}...")

for i in range(n_queries):
    # Модель выбирает самые "непонятные" данные
    query_idx, query_inst = learner.query(X_pool)
    
    # "Оракул" (мы) дает ответ
    learner.teach(X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(1,))
    
    # Удаляем из пула
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    
    if (i+1) % 10 == 0:
        y_pred = learner.predict(X_pool)
        f1 = f1_score(y_pool, y_pred)
        print(f"Query {i+1}/30: F1-score = {f1:.4f}")

# Финальный результат
y_pred_final = learner.predict(X_pool)
print("\nFinal Result:")
print(f"F1-score: {f1_score(y_pool, y_pred_final):.4f}")
