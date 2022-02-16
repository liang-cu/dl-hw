import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



n_epochs = 20

batch_size_train0 = 64
batch_size_train1 = 1024
batch_size_test = 64
learning_rate0 = 1e-3
learning_rate1 = 1e-2
momentum = 0.5
log_interval = 1000

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
  batch_size=batch_size_train0, shuffle=True)

train_loader1 = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train1, shuffle=True)

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
        self.fc1 = nn.Linear(320, 50) 
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
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


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
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
optimizer0 = optim.SGD(model0.parameters(), lr=learning_rate0,
                      momentum=momentum)
model1 = Net1()
optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate1,
                      momentum=momentum)

model2 = Net2()

     
#train model0
for epoch in range(1, n_epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader0):
      optimizer0.zero_grad()
      output = model0(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer0.step()
      if batch_idx % log_interval == 0:
         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader0.dataset),
            100. * batch_idx / len(train_loader0), loss.item()))
 
# train model1
for epoch in range(1, n_epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader1):
      optimizer1.zero_grad()
      output = model1(data)
      loss1 = F.nll_loss(output, target)
      loss1.backward()
      optimizer1.step()
      if batch_idx % log_interval == 0:
         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
         epoch, batch_idx * len(data), len(train_loader1.dataset),
          100. * batch_idx / len(train_loader1), loss1.item()))
 
alpha = torch.linspace(-1, 2, 30)
train_losses = [i for i in range(len(alpha))]
train_correct = [i for i in range(len(alpha))]
test_losses = [i for i in range(len(alpha))]
test_correct = [i for i in range(len(alpha))]
train_acc = [i for i in range(len(alpha))]
test_acc = [i for i in range(len(alpha))]

for i in range(len(alpha)):
    model0_parameter = nn.utils.parameters_to_vector(model0.parameters())
    model1_parameter = nn.utils.parameters_to_vector(model1.parameters())
    model2_parameter = ((1 - alpha[i]) * model0_parameter) + (alpha[i] * model1_parameter)
    list(model0.parameters())
    nn.utils.vector_to_parameters(model2_parameter, model2.parameters())
    for epoch in range(1, n_epochs + 1):
      with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader0):
          output = model2(data)
          loss2 = F.nll_loss(output, target)
          train_losses[i] = loss2.item()
          train_correct[i] += (output.argmax(1) == target).type(torch.float).sum().item()
    

    for epoch in range(1, n_epochs + 1):
      with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
          output2 = model2(data)
          t_loss2 = F.nll_loss(output2, target)
          test_losses[i] = t_loss2.item()
          test_correct[i] += (output2.argmax(1) == target).type(torch.float).sum().item()



train_acc = [correct / len(train_loader0.dataset) for correct in train_correct]
test_acc = [correct / len(test_loader.dataset) for correct in test_correct]


fig = plt.figure()
plt.plot(alpha, train_losses, color='blue')
plt.plot(alpha, test_losses, color='red')
plt.legend(['train loss','test loss'], loc='upper right')
plt.xlabel('alpha')
plt.ylabel('negative log likelihood loss')
fig
plt.show()


fig = plt.figure()
plt.plot(alpha, train_acc, color='blue')
plt.plot(alpha, test_acc, color='red')
plt.legend(['train accuracy','test accuracy'], loc='upper right')
plt.xlabel('alpha')
plt.ylabel('accuracy')
fig
plt.show()







