from __future__ import print_function
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from deep_emotion import Deep_Emotion
from data_loaders import Plain_Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = Plain_Dataset(csv_file='data/val.csv', img_dir='data/test/', datatype='test', transform=transformation)
test_loader = DataLoader(dataset, batch_size=64, num_workers=0)
model = 'deep_emotion-100-128-0.005.pt'
net = Deep_Emotion()
print("Deep Emotion:-", net)
net.load_state_dict(torch.load(model))
net.to(device)
net.eval()
# Model Evaluation on test data
classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
total = []

with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = net(data)
        pred = F.softmax(outputs, dim=1)
        classs = torch.argmax(pred, 1)
        wrong = torch.where(classs != labels, torch.tensor([1.]), torch.tensor([0.]))
        acc = 1 - (torch.sum(wrong) / 64)
        total.append(acc.item())

print('Accuracy of the network on the test images: %d %%' % (100 * np.mean(total)))

