import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


n_epochs = 20
batch_sizes = [32, 128, 512, 2048, 8192]
learning_rate = 1e-2
momentum = 0.5
log_interval = 10000

seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(seed)


###################load MNIST dataset####################
train_loader0 = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_sizes[0], shuffle=True)

train_loader1 = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_sizes[1], shuffle=True)

train_loader2 = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_sizes[2], shuffle=True)

train_loader3 = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_sizes[3], shuffle=True)

train_loader4 = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_sizes[4], shuffle=True)

test_loader0 = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_sizes[0], shuffle=True)

test_loader1 = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_sizes[1], shuffle=True)

test_loader2 = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_sizes[2], shuffle=True)

test_loader3 = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_sizes[3], shuffle=True)

test_loader4 = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_sizes[4], shuffle=True)

class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50) 
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


#model0 = Net0().to(device)
model0 = Net0()
model1 = Net0()
model2 = Net0()
model3 = Net0()
model4 = Net0()

for param in model1.parameters():
    param.requires_grad = True


optimizer0 = optim.SGD(model0.parameters(), lr=learning_rate,
                      momentum=momentum)
optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate,
                      momentum=momentum)
optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate,
                      momentum=momentum)
optimizer3 = optim.SGD(model3.parameters(), lr=learning_rate,
                      momentum=momentum)
optimizer4 = optim.SGD(model4.parameters(), lr=learning_rate,
                      momentum=momentum)

n_models = 5
train_losses = [i for i in range(n_models)]
test_losses = [i for i in range(n_models)]
train_correct = [i for i in range(n_models)]
train_acc = [i for i in range(n_models)]
test_correct = [i for i in range(n_models)]
test_acc = [i for i in range(n_models)]
test_acc = [i for i in range(n_models)]     
sensits =[i for i in range(n_models)]

#train model0
for epoch in range(1, n_epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader0):
      data.requires_grad_(True)
      optimizer0.zero_grad()
      output0 = model0(data)
      loss0 = F.nll_loss(output0, target)
      loss0.backward()
      optimizer0.step()
      train_losses[0] = loss0.item()
      train_correct[0] += (output0.argmax(1) == target).type(torch.float).sum().item()
      for p in model0.parameters():
        if p.grad is not None:
           grad0 = data.grad
           sens0 =  np.linalg.norm(grad0)
           #print("grad",grad0)
           #print('sens1',sens0)
      sensits[0] = sens0

      if batch_idx % log_interval == 0:
         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader0.dataset),
            100. * batch_idx / len(train_loader0), loss0.item()))
 
# train model1
for epoch in range(1, n_epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader1):
      optimizer1.zero_grad()
      data.requires_grad_(True)
      output1 = model1(data)
      loss1 = F.nll_loss(output1, target)
      loss1.backward()
      optimizer1.step()
      train_losses[1] = loss1.item()
      train_correct[1] += (output1.argmax(1) == target).type(torch.float).sum().item()
      #grad1 = torch.autograd.grad(loss1, data)
      #print("grad",grad1)
      for p in model1.parameters():
        if p.grad is not None:
           grad1 = data.grad
           sens1 =  np.linalg.norm(grad1)
           #print('sens1',sens1)
      sensits[1] = sens1


# train model2
for epoch in range(1, n_epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader2):
      data.requires_grad_(True)
      optimizer2.zero_grad()
      output2 = model2(data)
      loss2 = F.nll_loss(output2, target)
      loss2.backward()
      optimizer2.step()
      train_losses[2] = loss2.item()
      train_correct[2] += (output2.argmax(1) == target).type(torch.float).sum().item()
      for p in model2.parameters():
        if p.grad is not None:
           grad2 = data.grad
           sens2 =  np.linalg.norm(grad2)
      sensits[2] = sens2

# train model3
for epoch in range(1, n_epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader3):
      data.requires_grad_(True)
      optimizer3.zero_grad()
      output3 = model3(data)
      loss3 = F.nll_loss(output3, target)
      loss3.backward()
      optimizer3.step()
      train_losses[3] = loss3.item()
      train_correct[3] += (output3.argmax(1) == target).type(torch.float).sum().item()
      for p in model3.parameters():
        if p.grad is not None:
           grad3 = data.grad
           sens3 =  np.linalg.norm(grad3)
      sensits[3] = sens3

# train model4
for epoch in range(1, n_epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader4):
      data.requires_grad_(True)
      optimizer4.zero_grad()
      output4 = model4(data)
      loss4 = F.nll_loss(output4, target)
      loss4.backward()
      optimizer4.step()
      train_losses[4] = loss4.item()
      train_correct[4] += (output4.argmax(1) == target).type(torch.float).sum().item()
      for p in model4.parameters():
        if p.grad is not None:
           grad4 = data.grad
           sens4 =  np.linalg.norm(grad4)
           #print("grad",grad0)
           #print('sens1',sens0)
      sensits[4] = sens4

#test model0
for epoch in range(1, n_epochs + 1):
    for batch_idx, (data, target) in enumerate(test_loader0):
      output0 = model0(data)
      loss0 = F.nll_loss(output0, target)
      test_losses[0] = loss0.item()
      test_correct[0] += (output0.argmax(1) == target).type(torch.float).sum().item()

#test model1
for epoch in range(1, n_epochs + 1):
    for batch_idx, (data, target) in enumerate(test_loader1):
      output1 = model1(data)
      loss1 = F.nll_loss(output1, target)
      test_losses[1] = loss1.item()
      test_correct[1] += (output1.argmax(1) == target).type(torch.float).sum().item()

#test model2
for epoch in range(1, n_epochs + 1):
    for batch_idx, (data, target) in enumerate(test_loader2):
      output2 = model2(data)
      loss2 = F.nll_loss(output2, target)
      test_losses[2] = loss2.item()
      test_correct[2] += (output2.argmax(1) == target).type(torch.float).sum().item()
#test model3
for epoch in range(1, n_epochs + 1):
    for batch_idx, (data, target) in enumerate(test_loader3):
      output3 = model3(data)
      loss3 = F.nll_loss(output3, target)
      test_losses[3] = loss3.item()
      test_correct[3] += (output3.argmax(1) == target).type(torch.float).sum().item()

#test model4
for epoch in range(1, n_epochs + 1):
    for batch_idx, (data, target) in enumerate(test_loader4):
      output4 = model4(data)
      loss4 = F.nll_loss(output4, target)
      test_losses[4] = loss4.item()
      test_correct[4] += (output4.argmax(1) == target).type(torch.float).sum().item()



train_acc = [correct / len(train_loader0.dataset) for correct in train_correct]
test_acc = [correct / len(test_loader0.dataset) for correct in test_correct]

x = np.arange(0, len(batch_sizes))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(train_losses,'b', label = 'train_loss')
ax.plot(test_losses,'r', label = 'test_loss')
ax2 = ax.twinx()
ax2.plot(sensits,'g--', label = 'sensitivity')
ax.legend(loc=2)

ax.set_xlabel("batch size")
ax.set_ylabel("negative log likelihood loss")
ax2.set_ylabel("sensitivity")
ax2.legend(loc=1)
plt.xticks(x, batch_sizes)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(train_acc,'b', label = 'train_acc')
ax.plot(test_acc,'r', label = 'test_acc')
ax2 = ax.twinx()
ax2.plot(sensits,'g--', label = 'sensitivity')
ax.legend(loc=2)
ax.set_xlabel("batch size")
ax.set_ylabel("accuracy")
ax2.set_ylabel("sensitivity")
ax2.legend(loc=1)
plt.xticks(x, batch_sizes)
plt.show()




