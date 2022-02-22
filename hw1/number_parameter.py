import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n_epochs = 100

batch_size_train = 64
batch_size_test = 64
learning_rate = 0.01
momentum = 0.5
log_interval = 1000

seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(seed)

###################load MNIST dataset####################
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.fc1 = nn.Linear(500, 50) 
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)
        self.fc1 = nn.Linear(1000, 50)
        self.fc2 = nn.Linear(50, 10)
  
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, kernel_size=3)
        self.conv2 = nn.Conv2d(30, 60, kernel_size=3)
        self.fc1 = nn.Linear(1500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, kernel_size=3)
        self.conv2 = nn.Conv2d(40, 80, kernel_size=3)
        self.fc1 = nn.Linear(2000, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv1 = nn.Conv2d(1, 50, kernel_size=3)
        self.conv2 = nn.Conv2d(50, 100, kernel_size=3)
        self.fc1 = nn.Linear(2500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.conv1 = nn.Conv2d(1, 60, kernel_size=3)
        self.conv2 = nn.Conv2d(60, 120, kernel_size=3)
        self.fc1 = nn.Linear(3000, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        self.conv1 = nn.Conv2d(1, 70, kernel_size=3)
        self.conv2 = nn.Conv2d(70, 140, kernel_size=3)
        self.fc1 = nn.Linear(3500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class Net7(nn.Module):
    def __init__(self):
        super(Net7, self).__init__()
        self.conv1 = nn.Conv2d(1, 80, kernel_size=3)
        self.conv2 = nn.Conv2d(80, 160, kernel_size=3)
        self.fc1 = nn.Linear(4000, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net8(nn.Module):
    def __init__(self):
        super(Net8, self).__init__()
        self.conv1 = nn.Conv2d(1, 90, kernel_size=3)
        self.conv2 = nn.Conv2d(90, 180, kernel_size=3)
        self.fc1 = nn.Linear(4500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net9(nn.Module):
    def __init__(self):
        super(Net9, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, kernel_size=3)
        self.conv2 = nn.Conv2d(100, 200, kernel_size=3)
        self.fc1 = nn.Linear(5000, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model0 = Net0()
optimizer0 = optim.SGD(model0.parameters(), lr=learning_rate,
                      momentum=momentum)
model1 = Net1()
optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate,
                      momentum=momentum)
model2 = Net2()
optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate,
                      momentum=momentum)
model3 = Net3()
optimizer3 = optim.SGD(model3.parameters(), lr=learning_rate,
                      momentum=momentum)
model4 = Net4()
optimizer4 = optim.SGD(model4.parameters(), lr=learning_rate,
                      momentum=momentum)
model5 = Net5()
optimizer5 = optim.SGD(model5.parameters(), lr=learning_rate,
                      momentum=momentum)
model6 = Net6()
optimizer6 = optim.SGD(model6.parameters(), lr=learning_rate,
                      momentum=momentum)
model7 = Net7()
optimizer7 = optim.SGD(model7.parameters(), lr=learning_rate,
                      momentum=momentum)
model8 = Net8()
optimizer8 = optim.SGD(model8.parameters(), lr=learning_rate,
                      momentum=momentum)
model9 = Net9()
optimizer9 = optim.SGD(model9.parameters(), lr=learning_rate,
                      momentum=momentum)
                      
print('model0 paramters', count_parameters(model0))
print('model1 paramters', count_parameters(model1))
print('model2 paramters', count_parameters(model2))
print('model3 paramters', count_parameters(model3))
print('model4 paramters', count_parameters(model4))
print('model5 paramters', count_parameters(model5))
print('model6 paramters', count_parameters(model6))
print('model7 paramters', count_parameters(model7))
print('model8 paramters', count_parameters(model8))
print('model9 paramters', count_parameters(model9))

n_models = 10
train_losses = [i for i in range(n_models)]
test_losses = [i for i in range(n_models)]
train_correct = [i for i in range(n_models)]
train_acc = [i for i in range(n_models)]
test_correct = [i for i in range(n_models)]
test_acc = [i for i in range(n_models)]
n_parameter = []


models = [model0, model1, model2, model3, model4, model5, model6, model7, model8, model9]

for model in models:
    n_parameter.append(count_parameters(model))


for epoch in range(1, n_epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer0.zero_grad()
      optimizer1.zero_grad()
      optimizer2.zero_grad()
      optimizer3.zero_grad()
      optimizer4.zero_grad()
      optimizer5.zero_grad()
      optimizer6.zero_grad()
      optimizer7.zero_grad()
      optimizer8.zero_grad()
      optimizer9.zero_grad()

      output0 = model0(data)
      output1 = model1(data)
      output2 = model2(data)
      output3 = model3(data)
      output4 = model4(data)
      output5 = model5(data)
      output6 = model6(data)
      output7 = model7(data)
      output8 = model8(data)
      output9 = model9(data)

      loss0 = F.nll_loss(output0, target)
      loss1 = F.nll_loss(output1, target)
      loss2 = F.nll_loss(output2, target)
      loss3 = F.nll_loss(output3, target)
      loss4 = F.nll_loss(output4, target)
      loss5 = F.nll_loss(output5, target)
      loss6 = F.nll_loss(output6, target)
      loss7 = F.nll_loss(output7, target)
      loss8 = F.nll_loss(output8, target)
      loss9 = F.nll_loss(output9, target)
      
      train_losses[0] = loss0.item()   
      train_losses[1] = loss1.item()
      train_losses[2] = loss2.item()
      train_losses[3] = loss3.item()
      train_losses[4] = loss4.item()
      train_losses[5] = loss5.item()
      train_losses[6] = loss6.item()
      train_losses[7] = loss7.item()
      train_losses[8] = loss8.item()
      train_losses[9] = loss9.item()      
      
      loss0.backward()
      loss1.backward()
      loss2.backward()
      loss3.backward()
      loss4.backward()
      loss5.backward()        
      loss6.backward()
      loss7.backward()
      loss8.backward()
      loss9.backward()

      optimizer0.step()
      optimizer1.step()
      optimizer2.step()
      optimizer3.step()
      optimizer4.step()
      optimizer5.step()
      optimizer6.step()
      optimizer7.step()
      optimizer8.step()
      optimizer9.step()

      train_correct[0] += (output0.argmax(1) == target).type(torch.float).sum().item()
      train_correct[1] += (output1.argmax(1) == target).type(torch.float).sum().item()
      train_correct[2] += (output2.argmax(1) == target).type(torch.float).sum().item()
      train_correct[3] += (output3.argmax(1) == target).type(torch.float).sum().item()
      train_correct[4] += (output4.argmax(1) == target).type(torch.float).sum().item()
      train_correct[5] += (output5.argmax(1) == target).type(torch.float).sum().item()
      train_correct[6] += (output6.argmax(1) == target).type(torch.float).sum().item()
      train_correct[7] += (output7.argmax(1) == target).type(torch.float).sum().item()
      train_correct[8] += (output8.argmax(1) == target).type(torch.float).sum().item()
      train_correct[9] += (output9.argmax(1) == target).type(torch.float).sum().item()
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
         epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss0.item()))

train_acc = [correct / len(train_loader.dataset) for correct in train_correct]


for epoch in range(1, n_epochs + 1):
   with torch.no_grad():
       for batch_idx, (data, target) in enumerate(test_loader):
         output0 = model0(data)
         output1 = model1(data)
         output2 = model2(data)
         output3 = model3(data)
         output4 = model4(data)
         output5 = model5(data)
         output6 = model6(data)
         output7 = model7(data)
         output8 = model8(data)
         output9 = model9(data)

         t_loss0 = F.nll_loss(output0, target)
         t_loss1 = F.nll_loss(output1, target)
         t_loss2 = F.nll_loss(output2, target)
         t_loss3 = F.nll_loss(output3, target)
         t_loss4 = F.nll_loss(output4, target)
         t_loss5 = F.nll_loss(output5, target)
         t_loss6 = F.nll_loss(output6, target)
         t_loss7 = F.nll_loss(output7, target)
         t_loss8 = F.nll_loss(output8, target)
         t_loss9 = F.nll_loss(output9, target)
         
         test_losses[0] = t_loss0.item()
         test_losses[1] = t_loss1.item()
         test_losses[2] = t_loss2.item()
         test_losses[3] = t_loss3.item()
         test_losses[4] = t_loss4.item()
         test_losses[5] = t_loss5.item()
         test_losses[6] = t_loss6.item()
         test_losses[7] = t_loss7.item()
         test_losses[8] = t_loss8.item()
         test_losses[9] = t_loss9.item()


         test_correct[0] += (output0.argmax(1) == target).type(torch.float).sum().item()
         test_correct[1] += (output1.argmax(1) == target).type(torch.float).sum().item()
         test_correct[2] += (output2.argmax(1) == target).type(torch.float).sum().item()
         test_correct[3] += (output3.argmax(1) == target).type(torch.float).sum().item()
         test_correct[4] += (output4.argmax(1) == target).type(torch.float).sum().item()
         test_correct[5] += (output5.argmax(1) == target).type(torch.float).sum().item()
         test_correct[6] += (output6.argmax(1) == target).type(torch.float).sum().item()
         test_correct[7] += (output7.argmax(1) == target).type(torch.float).sum().item()
         test_correct[8] += (output8.argmax(1) == target).type(torch.float).sum().item()
         test_correct[9] += (output9.argmax(1) == target).type(torch.float).sum().item()

test_acc = [correct / len(test_loader.dataset) for correct in test_correct]




fig = plt.figure()
plt.plot(n_parameter, train_losses, color='blue')
plt.plot(n_parameter, test_losses, color='red')
plt.legend(['train loss','test loss'], loc='upper right')
plt.xlabel('number of parameters')
plt.ylabel('negative log likelihood loss')
fig
plt.show()


fig = plt.figure()
plt.plot(n_parameter, train_acc, color='blue')
plt.plot(n_parameter, test_acc, color='red')
plt.legend(['train accuracy','test accuracy'], loc='upper right')
plt.xlabel('number of parameters')
plt.ylabel('accuracy')
fig
plt.show()


