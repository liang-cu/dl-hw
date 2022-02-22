import math
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
from autograd_lib import autograd_lib
from collections import defaultdict

n_epochs = 100
learning_rate = 0.01
momentum = 0.5
seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(seed)

x = torch.linspace(0.1*math.pi, 3*math.pi, 2000)
y = torch.sin(x)/x


x = x.reshape((len(x), 1))
y = y.reshape((len(y), 1))

class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.f1 = nn.Linear(1, 5)
        self.f2 = nn.Linear(5, 10)
        self.f3 = nn.Linear(10, 10)
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = F.relu(self.f3(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

loss_fn = torch.nn.MSELoss(reduction='sum')
model = Net0()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)
train_losses = []
train_loss1 = []
accuracy0 = []
weights_list = []
grad_norm_all = []
test_losses = []


def train1():
   model = Net0()
   for  epoch in range(1, n_epochs + 1):
        output = model(x)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
   return model


# model0 train
activations = {}

def save_activations(layer, A, _):
    activations[layer] = A
def compute_hess(layer, _, B):
    A = activations[layer]
    BA = torch.einsum("nl,ni->nli", B, A)

for epoch in range(1, n_epochs + 1):
    trainCorrect0 = 0
    activations = defaultdict(int)
    hess = defaultdict(float)
    model1 = train1()
    autograd_lib.register(model1)
    model1.train()
    model1.zero_grad()
    output = model1(x)
    loss = loss_fn(output,y)
    loss.backward()
    
    grad_all = 0.0
    for p in model1.modules():
        if isinstance(p,nn.Linear):
           grad_norm = p.weight.grad.norm(2).item()
           grad_norm_all.append(grad_norm)
    grad_norm_mean = np.mean(grad_norm_all) 
    model1.zero_grad()
    with autograd_lib.module_hook(save_activations):
        output1 = model1(x)
        loss1 = loss_fn(output1, y)
    with autograd_lib.module_hook(compute_hess):
        autograd_lib.backward_hessian(output1, loss='LeastSquares')
    layer_hess = list(hess.values())

    mini_ratio = []
    for hess in layer_hess:
      length = hess.shape[0] * hess.shape[1]
      hess = hess.reshape(length,length)
      eigen_val = torch.symeig(hess).eigenvalues
      num = torch.sum(eigen_val > 0).item()
      mini_ratio.append(num / len(eigen_val))
    ratio_mean = np.mean(mini_ratio)


 
print("grad",grad_norm_mean)
print("loss",loss)
print("ratio",ratio_mean)
#
#    else:
#        grad_all = 0.0
#        for p in model0.parameters():
#           grad = 0.0
#           if p.grad is not None:
#               grad = (p.grad.cpu().data.numpy() ** 2).sum()
#               grad_norm_all.append(grad_norm)
##        print("grad norm", grad_norm) 
#        loss2 = torch.from_numpy(np.asarray(grad_norm))
#        loss2.requires_grad_(True)
#        loss = loss2
#     
#
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
# 
#      #grad_norm_all.append(grad_norm)
#      train_losses.append(loss.item())
#      trainCorrect0 += (output.argmax(1) == target).type(torch.float).sum().item()
#      trainCorrect0 = trainCorrect0 / len(train_loader.dataset)
#
#      if batch_idx % log_interval == 0:
#         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#            epoch, batch_idx * len(data), len(train_loader.dataset),
#            100. * batch_idx / len(train_loader), loss.item()))
## plot gradient and loss
#fig, ax = plt.subplots(2)
#ax[0].plot( grad_norm_all, color='blue')
#ax[1].plot( train_losses, color= 'blue')
#ax[0].set_xlabel('iteration')
#ax[0].set_ylabel('grad')
#ax[1].set_xlabel('iteration')
#ax[1].set_ylabel('loss')
#plt.show()
#
