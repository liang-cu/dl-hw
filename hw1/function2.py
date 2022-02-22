import torch
import math
import matplotlib.pyplot as plt
import numpy 
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

loss_values0 = []
weights_list = []
grad_norm_all = []
learning_rate = 1e-5
momentum = 0.5

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# Create Tensors to hold input and outputs.

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
        x = self.f3(x)
        x = self.f3(x)
        x = self.f3(x)
        x = F.relu(self.f3(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x


#model0 = Net0().to(device)
model0 = Net0()
print(model0)
optimizer = optim.SGD(model0.parameters(), lr=learning_rate,
                      momentum=momentum)


print('model0',model0)

print('model0 paramters', count_parameters(model0))

# use Mean Squared Error (MSE) as loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')



for t in range(10000):
   
    y_pred0 = model0(x)
    loss0 = loss_fn(y_pred0, y)

    #if t % 100 == 99:
    #    print(t, loss0.item(),'\t',loss1.item(),'\t', loss2.item())


    # Zero the gradients before running the backward pass.
    model0.zero_grad()
    loss0.backward()
    


    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients

    with torch.no_grad():
        for param0 in model0.parameters():
            param0 -= learning_rate * param0.grad

    
    if t % 100 == 99:
        print(t, loss0.item())
       # loss_values0.append(loss0.item())


    grad_all = 0.0


    for p in model0.parameters():
         grad = 0.0
         if p.grad is not None:
             grad = (p.grad.cpu().data.numpy() ** 2).sum()
         grad_all += grad
    grad_norm = grad_all ** 0.5
    grad_norm_all.append(grad_norm)
    loss_values0.append(loss0.item())



    if t % 3 == 0:
       #weight = nn.utils.parameters_to_vector(model0.parameters())
       weight = torch.squeeze(model0.f2.weight)
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
        ax.scatter(principalComponents[:,0], principalComponents[:,1],   alpha=0.5)
        #ax.scatter(principalComponents[:,0], principalComponents[:,1], label=w.shape, alpha=0.5)
    ax.legend()
    plt.show()



plot_weights(weights_list)
#plot_fc_weights(data_2d)




#print("model0.weight",model0.layer[0].weight) 

#pca = PCA(n_components=2)
#pca.fit(X)
#print(pca.components_)

#print(loss_values0)


fig, ax = plt.subplots(2)
ax[0].plot( grad_norm_all, color='blue')
ax[1].plot( loss_values0, color= 'blue')
ax[0].set_xlabel('iteration')
ax[0].set_ylabel('grad')
ax[1].set_xlabel('iteration')
ax[1].set_ylabel('loss')
plt.show()


