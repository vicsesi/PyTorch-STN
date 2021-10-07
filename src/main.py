import torch
import argparse
from net import Net
import torch.optim as optim
from six.moves import urllib
from utils import train, test, get_train_loader
from utils import get_test_loader, visualize_stn

parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=str)
parser.add_argument('--epochs', type=int)
args = parser.parse_args()

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_train_loader()
test_loader = get_test_loader()
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

if 'train' in args.stage:
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, device, optimizer)

if 'test' in args.stage:
    test(model, test_loader, device)
    visualize_stn(model, test_loader, device)
