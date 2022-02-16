import random
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor



n_epochs = 1000

batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.1
momentum = 0.5
log_interval = 1000

seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(seed)


###################load MNIST dataset####################
train = datasets.MNIST(
    root='./files/',
    train=True,
    download=True,
    transform=ToTensor()
)

test = datasets.MNIST(
    root='./files/',
    train=False,
    download=True,
    transform=ToTensor()
)

## shuffle train labels
#print('targets',train.targets)
random.shuffle(train.targets)
#print('random',train.targets)
train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=1000, shuffle=True)

class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50) 
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.reshape(x.shape[0], -1)
        #x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

#model0 = Net0().to(device)
model0 = Net0()
print(model0)
optimizer = optim.SGD(model0.parameters(), lr=learning_rate,
                      momentum=momentum)



train_losses = []
train_losses1 = []
train_losses2 = []
train_counter = []
train_counter1 = []
train_counter2 = []
accuracy0 = []
accuracy1 = []
accuracy2 = []


test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


#def train(epoch):
#  trainCorrect0 = 0
#  model0.train()
#  for batch_idx, (data, target) in enumerate(train_loader):
#    optimizer.zero_grad()
#    output = model0(data)
#    loss = F.nll_loss(output, target)
#    loss.backward()
#    trainCorrect0 += (output.argmax(1) == target).type(torch.float).sum().item()
#      #print ('train correct', trainCorrect)
#    trainCorrect0 = trainCorrect0 / len(train_loader.dataset)
#
#    optimizer.step()
#    if batch_idx % log_interval == 0:
#      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#        epoch, batch_idx * len(data), len(train_loader.dataset),
#        100. * batch_idx / len(train_loader), loss.item()))
#    train_losses.append(loss.item())
#    train_counter.append(
#        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
#    train_accuracy0.append(100.*trainCorrect0)
#


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
    trainCorrect0 = 0
    avg_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
      #data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      output = model0(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      avg_loss+=F.nll_loss(output, target, reduction='sum').item()
      

      
      if batch_idx % log_interval == 0:
         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
      

    
    accur = test()
    train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
    #train_accuracy0.append(100.*trainCorrect0)
    avg_loss/=len(train_loader.dataset)
    train_losses.append(avg_loss)
    accuracy0.append(accur)



 

fig = plt.figure()
plt.plot( train_losses, color='blue')
plt.plot( test_losses, color='red')

plt.legend(['train loss','test loss'], loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('negative log likelihood loss')
fig
plt.show()

