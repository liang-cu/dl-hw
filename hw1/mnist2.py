import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 1000

seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(seed)

#device = torch.device("cuda" if use_cuda else "cpu")
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


#model0 = Net0().to(device)
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

# gradient
      grad_all = 0.0


      for p in model0.parameters():
           grad = 0.0
           if p.grad is not None:
               grad = (p.grad.cpu().data.numpy() ** 2).sum()
           grad_all += grad
      grad_norm = grad_all ** 0.5
      #print("grad",grad_norm)
      grad_norm_all.append(grad_norm)
      train_losses.append(loss.item())

      
      if batch_idx % log_interval == 0:
         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
    

    #get weights and reduce dimensions

    if epoch % 1 == 0:
       #weight = nn.utils.parameters_to_vector(model0.parameters())
       weight = torch.squeeze(model0.conv1.weight)
       weight = weight.detach().numpy()
       #nx, ny, nz = weight.shape
       #weights_list.append(weight.reshape(nx*ny, nz))
       #weights_list.append(weight)

       data_2d = [features_2d.flatten() for features_2d in weight]
       weights_list.append(data_2d)

      


def plot_weights(weights_list):
    pca = PCA(n_components=2)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('2 component PCA')
    for w in weights_list:      
        #print(w.shape)
        #w = w.reshape(-1, 1)
        #principalComponents = pca.fit_transform(w.detach().numpy())
        principalComponents = pca.fit_transform(w)
        ax.scatter(principalComponents[:,0], principalComponents[:,1], label= (np.array(w)).shape,  alpha=0.5)
        #ax.scatter(principalComponents[:,0], principalComponents[:,1], label=w.shape, alpha=0.5)
    ax.legend()
    plt.show()



plot_weights(weights_list)
#plot_fc_weights(data_2d)



# plot gradient and loss
fig, ax = plt.subplots(2)
ax[0].plot( grad_norm_all, color='blue')
ax[1].plot( train_losses, color= 'blue')
ax[0].set_xlabel('iteration')
ax[0].set_ylabel('grad')
ax[1].set_xlabel('iteration')
ax[1].set_ylabel('loss')
plt.show()

