import torch
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from model import RespiratoryTransformer


classes = ["URT", "ILD", "Asthma", "COPD"]


def generate_dummy_data(n=200, seq_len=10, input_dim=64):
    X = torch.randn(n, seq_len, input_dim)
    y = torch.randint(0, 4, (n,))
    return X, y


def evaluate():
    model = RespiratoryTransformer()
    model.eval()

    X, y_true = generate_dummy_data()

    with torch.no_grad():
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)

    y_true_np = y_true.numpy()
    preds_np = preds.numpy()

    print("\n📊 Confusion Matrix")
    print(confusion_matrix(y_true_np, preds_np))

    print("\n📄 Classification Report")
    print(classification_report(y_true_np, preds_np, target_names=classes))


if __name__ == "__main__":
    evaluate()
