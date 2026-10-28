import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Подготавливаем изображения
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data, batch_size=64, shuffle=True
)
test_loader = DataLoader(
    test_data, batch_size=64
)

# 2. Создаём архитектуру нейросети
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


model = NeuralNetwork()

# 3. Настраиваем обучение
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

# 4. Обучаем нейросеть
for epoch in range(3):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        predictions = model(images)
        loss = loss_function(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Эпоха {epoch + 1}/3, "
        f"ошибка: {total_loss / len(train_loader):.4f}"
    )

# 5. Проверяем точность
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images)
        predicted_labels = predictions.argmax(dim=1)

        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Точность на тестовых данных: {accuracy:.2f}%")

# 6. Сохраняем обученную модель
torch.save(model.state_dict(), "model.pth")
print("Модель сохранена в model.pth")
