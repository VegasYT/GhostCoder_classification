import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle


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

data = pd.read_csv('dataset.csv')

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data['text'].fillna('', inplace=True)

label_encoder = LabelEncoder()

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['text'])
X_test = vectorizer.transform(test_data['text'])

train_labels_encoded = label_encoder.fit_transform(train_data['class'])
test_labels_encoded = label_encoder.transform(test_data['class'])

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.toarray(), dtype=torch.float32)
        self.y = torch.tensor(y.astype(int), dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CustomDataset(X_train, train_labels_encoded)
test_dataset = CustomDataset(X_test, test_labels_encoded)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

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

input_size = X_train.shape[1]
hidden_size = 128 # Размер скрытого слоя
num_classes = len(data['class'].unique())
print(num_classes)
model = TextClassifier(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")
torch.save(model.state_dict(), 'model.pth')

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

class_df = pd.DataFrame({
    'class': list(range(len(label_encoder.classes_))),
    'name': [classses.get(class_name, class_name) for class_name in label_encoder.classes_]
})

class_df.to_csv('class.csv', index=False)