import torch
import torch.nn as nn
import torch.optim as optim


class SquareNet(nn.Module):
    def __init__(self):
        super(SquareNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# Создаём данные: x от -10 до 10, y = x^2
x = torch.linspace(0, 50, 100000).unsqueeze(1)
y = x ** 2

# Инициализируем модель, loss и оптимайзер
model = SquareNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Обучение
epochs = 2000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

torch.save(model.state_dict(), "model.pt")