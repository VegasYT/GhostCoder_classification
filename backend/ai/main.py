import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class TextClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)  # Добавляем dropout с вероятностью 0.5
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # Добавляем скрытый слой с активацией ReLU
        x = self.dropout(x)  # Применяем dropout
        x = self.fc2(x)
        return x

classses = {
    'act': 'акт',
    'proxy': 'доверенность',
    'contract': 'договор',
    'application': 'заявление',
    'order': 'приказ',
    'invoice': 'счет',
    'bill': 'приложение',
    'arrangement': 'соглашение',
    'contract offer': 'договор оферты',
    'statute': 'устав',
    'determination': 'решение'
}

def distribution(text: str):
    df = pd.read_csv('ai/class.csv')
    label = predict(text)
    for _, row in df.iterrows():
        if row['class'] == label:
            return row['name']

def predict(text: str) -> int:
    vectorizer_path = 'ai/vectorizer.pkl'
    model_path = 'ai/model.pth'

    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    text_to_classify = vectorizer.transform([text])

    input_size = text_to_classify.shape[1]
    hidden_size = 128  # Размер скрытого слоя
    num_classes = 11
    model = TextClassifier(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        outputs = model(torch.tensor(text_to_classify.toarray(), dtype=torch.float32))

    # Не применяем softmax, получаем просто выход модели
    predicted_class = torch.argmax(outputs, dim=1).item()

    # Применяем softmax только для вывода вероятностей
    probabilities = F.softmax(outputs, dim=1)
    percentages = [prob * 100 for prob in probabilities.squeeze().tolist()]

    print("Predicted class:", predicted_class)
    for i, percent in enumerate(percentages):
        print(f"Class {i}: {percent:.2f}%")

    return predicted_class