import torch
from model import RespiratoryTransformer


# کلاس‌ها
classes = ["URT", "ILD", "Asthma", "COPD"]


def load_model():
    model = RespiratoryTransformer()
    model.eval()
    return model


def predict(model, sample):
    """
    sample shape: (1, seq_len, input_dim)
    """

    with torch.no_grad():
        output = model(sample)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    return classes[pred_class], probs.numpy()


if __name__ == "__main__":

    # نمونه ورودی ساختگی (مثل patient data)
    sample = torch.randn(1, 10, 64)

    model = load_model()

    prediction, probabilities = predict(model, sample)

    print("\n🧠 Prediction Result")
    print("---------------------")
    print("Predicted Disease:", prediction)
    print("Probabilities:", probabilities)
