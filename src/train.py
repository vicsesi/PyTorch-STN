import torch
import numpy as np
import torchvision
from net import Net
import torch.optim as optim
from six.moves import urllib
from utils import train, test
import matplotlib.pyplot as plt
from utils import get_train_loader, get_test_loader

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_train_loader()
test_loader = get_test_loader()
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1, 2 + 1):
    train(epoch, model, train_loader, device, optimizer)
    test(model, test_loader, device)

visualize_stn(model, device)
plt.ioff()
plt.show()
