import torch
import torch.nn as nn
import torch.optim as optim
from model import RespiratoryTransformer


# -----------------------
# Fake dataset (for now)
# -----------------------
def generate_dummy_data(num_samples=500, seq_len=10, input_dim=64, num_classes=4):
    X = torch.randn(num_samples, seq_len, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


# -----------------------
# Training function
# -----------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RespiratoryTransformer()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X, y = generate_dummy_data()
    X, y = X.to(device), y.to(device)

    epochs = 10

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()

        outputs = model(X)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        # accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y).float().mean()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f} Accuracy: {acc.item():.4f}")


if __name__ == "__main__":
    train()
