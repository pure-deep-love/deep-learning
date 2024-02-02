import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
import PIL

train_data = torchvision.datasets.MNIST(root='../data', train=True,
                                         transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST(root='../data', train=False, 
                                       transform=torchvision.transforms.ToTensor(), download=True)
train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(), 
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 10))
net = net.cuda()

loss = nn.CrossEntropyLoss(reduction='none').cuda()
trainer = torch.optim.SGD(net.parameters(), lr=1e-2)

num_epochs = 10
for epoch in range(num_epochs):
    l_sum, train_acc_num = 0, 0
    net.train()
    for X, y in train_dataloader:
        X, y = X.cuda(), y.cuda()
        y_hat = net(X)
        l = loss(y_hat, y)
        trainer.zero_grad()
        train_acc_num  += (y_hat.argmax(axis=1) == y).sum()
        l.mean().backward()
        trainer.step()
        l_sum += l.sum()
    print(f'epoch {epoch}:\ntrain loss mean: {l_sum / len(train_data):.2f}')
    print(f'train accuracy: {train_acc_num / len(train_data):.2f}')

    test_acc_num, test_l_sum = 0, 0
    net.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.cuda(), y.cuda()
            y_hat = net(X)
            l = loss(y_hat, y)
            test_acc_num += (y_hat.argmax(axis=1) == y).sum()
            test_l_sum += l.sum()
    print(f'test loss mean: {test_l_sum / len(test_data):.2f}')
    print(f'test accuracy: {test_acc_num / len(test_data):.2f}')

torch.save(net.state_dict(), '../state_dict/han_num.params')