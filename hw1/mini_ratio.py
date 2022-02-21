import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA

n_epochs = 20
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 1000
seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(seed)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)


class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50) #for MNIST dataset
        #self.fc1 = nn.Linear(500, 50)# for CIFAR-10 dataset
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.reshape(x.shape[0], -1)
        #x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


model0 = Net0()
print(model0)
optimizer = optim.SGD(model0.parameters(), lr=learning_rate,
                      momentum=momentum)
train_losses = []
train_counter = []
accuracy0 = []
weights_list = []
grad_norm_all = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def test():
  model0.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = model0(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  accuracy = 100. * correct / len(test_loader.dataset)
  return accuracy


# model0 train

for epoch in range(1, n_epochs + 1):
#    accur = test()
    trainCorrect0 = 0
    for batch_idx, (data, target) in enumerate(train_loader):
      output = model0(data)
      loss1 = F.nll_loss(output, target)
      if epoch < 2:
        loss = loss1            
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if epoch >=2:
        grad_all = 0.0
        for p in model0.parameters():
           grad = 0.0
           if p.grad is not None:
               grad = (p.grad.cpu().data.numpy() ** 2).sum()
           grad_all += grad
        grad_norm = grad_all ** 0.5
        print("grad norm", grad_norm) 
        loss2 = torch.from_numpy(np.asarray(grad_norm))
        loss2.requires_grad_(True)
        loss = loss2
     

      #if epoch < 10:
      #  loss = loss1
      #else:
      #  loss = loss2
 
        grad_norm_all.append(grad_norm)
        train_losses.append(loss.item())
        trainCorrect0 += (output.argmax(1) == target).type(torch.float).sum().item()
        trainCorrect0 = trainCorrect0 / len(train_loader.dataset)

      if batch_idx % log_interval == 0:
         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
# plot gradient and loss
fig, ax = plt.subplots(2)
ax[0].plot( grad_norm_all, color='blue')
ax[1].plot( train_losses, color= 'blue')
ax[0].set_xlabel('iteration')
ax[0].set_ylabel('grad')
ax[1].set_xlabel('iteration')
ax[1].set_ylabel('loss')
plt.show()

