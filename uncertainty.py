import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms

from model import MC_Dropout_Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

model = MC_Dropout_Net().to(device)

model.load_state_dict(
    torch.load("mc_dropout_mnist.pth")
)

# IMPORTANT!
# Keep dropout active
model.train()

image, label = dataset[0]

image = image.unsqueeze(0).to(device)

predictions = []

T = 30

for _ in range(T):

    output = model(image)

    probability = torch.softmax(output, dim=1)

    predictions.append(
        probability.detach().cpu().numpy()
    )

predictions = np.array(predictions)

mean_prediction = predictions.mean(axis=0)

std_prediction = predictions.std(axis=0)

predicted_class = np.argmax(mean_prediction)

confidence = np.max(mean_prediction)

uncertainty = np.mean(std_prediction)

print("True Label:", label)
print("Prediction:", predicted_class)
print("Confidence:", confidence)
print("Uncertainty:", uncertainty)
