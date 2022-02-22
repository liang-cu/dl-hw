import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



n_epochs = 10

batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 1000

seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(seed)

#device = torch.device("cuda" if use_cuda else "cpu")
####################load MNIST dataset####################
#train_loader = torch.utils.data.DataLoader(
#  torchvision.datasets.MNIST('./files/', train=True, download=True,
#                             transform=torchvision.transforms.Compose([
#                               torchvision.transforms.ToTensor(),
#                               torchvision.transforms.Normalize(
#                                 (0.1307,), (0.3081,))
#                             ])),
#  batch_size=batch_size_train, shuffle=True)
#
#test_loader = torch.utils.data.DataLoader(
#  torchvision.datasets.MNIST('./files/', train=False, download=True,
#                             transform=torchvision.transforms.Compose([
#                               torchvision.transforms.ToTensor(),
#                               torchvision.transforms.Normalize(
#                                 (0.1307,), (0.3081,))
#                             ])),
#  batch_size=batch_size_test, shuffle=True)

####################load CIFAR-10 dataset####################

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.fc1 = nn.Linear(320, 50) #for MNIST dataset
        self.fc1 = nn.Linear(500, 50)# for CIFAR-10 dataset
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.reshape(x.shape[0], -1)
        #x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv4 = nn.Conv2d(40, 80, kernel_size=3)
        #self.fc1 = nn.Linear(80, 10)# for MNIST dataset
        self.fc1 = nn.Linear(320, 10)# for CIFAR dataset

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        #x = F.dropout(x, training=self.training)
        return F.log_softmax(x)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv5 = nn.Conv2d(128,256, kernel_size=3)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3)

        #self.fc1 = nn.Linear(2048, 10) # for MNIST
        self.fc1 = nn.Linear(4608, 10) # for CIFAR
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(F.max_pool2d(self.conv6(x),2))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x)


#model0 = Net0().to(device)
model0 = Net0()
print(model0)
optimizer = optim.SGD(model0.parameters(), lr=learning_rate,
                      momentum=momentum)


model1 = Net1()
print(model1)
optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate,
                      momentum=momentum)

model2 = Net2()
print(model2)
optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate,
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

def test1():
  model1.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = model1(data)
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

def test2():
  model2.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = model2(data)
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
    accur = test()
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
      trainCorrect0 += (output.argmax(1) == target).type(torch.float).sum().item()
 #     print ('train correct', trainCorrect0)
      trainCorrect0 = trainCorrect0 / len(train_loader.dataset)
#      print ('train correct', trainCorrect0)

      
      if batch_idx % log_interval == 0:
         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
      

    

    train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
    #train_accuracy0.append(100.*trainCorrect0)
    avg_loss/=len(train_loader.dataset)
    train_losses.append(avg_loss)
    accuracy0.append(accur)



 
# model 1 train
for epoch in range(1, n_epochs + 1):
    avg_loss1 = 0
    accur1 = test1()
    trainCorrect1 = 0
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer1.zero_grad()
      output = model1(data)
      loss1 = F.nll_loss(output, target)
      loss1.backward()
      trainCorrect1 += (output.argmax(1) == target).type(torch.float).sum().item()
      #print ('train correct', trainCorrect)
      trainCorrect1 = trainCorrect1 / len(train_loader.dataset)
      avg_loss1+=F.nll_loss(output, target, reduction='sum').item()
      optimizer1.step()
      if batch_idx % log_interval == 0:
         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
         epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss1.item()))
      #train_losses1.append(loss1.item())
      #train_counter1.append(
      #      (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

   
    train_counter1.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
    #train_accuracy0.append(100.*trainCorrect0)
    avg_loss1/=len(train_loader.dataset)
    train_losses1.append(avg_loss1)
    accuracy1.append(accur1)













#model2 train 
for epoch in range(1, n_epochs + 1):
    trainCorrect2 = 0
    avg_loss2 = 0
    accur2 = test2()
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer2.zero_grad()
      output = model2(data)
      loss2 = F.nll_loss(output, target)
      loss2.backward()
      avg_loss2+=F.nll_loss(output, target, reduction='sum').item()
      trainCorrect2 += (output.argmax(1) == target).type(torch.float).sum().item()
      #print ('train correct', trainCorrect)
      trainCorrect2 = trainCorrect2 / len(train_loader.dataset)

      optimizer2.step()
      if batch_idx % log_interval == 0:
         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
         epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss2.item()))
      #train_losses2.append(loss2.item())
      #train_counter2.append(
      #      (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      #train_accuracy2.append(trainCorrect2)
    avg_loss2/=len(train_loader.dataset)
    train_losses2.append(avg_loss2)
    accuracy2.append(accur2)









fig = plt.figure()
plt.plot( train_losses, color='blue')
plt.plot( train_losses1, color='red')
plt.plot( train_losses2, color='black')
plt.legend(['Model0 train Loss','Model1 train loss','Model2 train loss'], loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('negative log likelihood loss')
fig
plt.show()


fig = plt.figure()
plt.plot( accuracy0, color='blue')
plt.plot( accuracy1, color='red')
plt.plot( accuracy2, color='black')
plt.legend(['Model0 train accuracy','Model1 train accuracy','Model2 train accuracy'], loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Accucary')
fig
plt.show()

