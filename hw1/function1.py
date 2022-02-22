import torch
import math
import matplotlib.pyplot as plt
import numpy 


loss_values0 = []
loss_values1 = []
loss_values2 = []
grad_norm0 = []

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Create Tensors to hold input and outputs.
# for sin(x) simulation 
#x = torch.linspace(-math.pi, math.pi, 2000)
#y = torch.sin(x)

# for sin(x)/x simulation 
x = torch.linspace(0.1*math.pi, 3*math.pi, 2000)
y = torch.sin(x)/x

x = x.reshape((len(x), 1))
y = y.reshape((len(y), 1))

model0 = torch.nn.Sequential(
    torch.nn.Linear(1, 5),
    torch.nn.Linear(5, 10),
    torch.nn.Linear(10, 10),
    torch.nn.Linear(10, 10),
    torch.nn.Linear(10, 10),
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,5),
    torch.nn.Linear(5,1)


)


model1 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.Linear(10, 18),
    torch.nn.Linear(18, 15),
    torch.nn.Linear(15, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4,1)
   
)


model2 = torch.nn.Sequential(
    torch.nn.Linear(1, 190),
    torch.nn.ReLU(),
    torch.nn.Linear(190,1)

)


print('model0',model0)
print('model1',model1)
print('model2',model2)

#for parameter in model0.parameters():
#    print('model0 parameter',parameter)
print('model0 paramters', count_parameters(model0))
print('model1 paramters', count_parameters(model1))
print('model2 paramters', count_parameters(model2))

# use Mean Squared Error (MSE) as loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')
#learning_rate0 = 8e-6
#learning_rate1 = 8e-6
#learning_rate2 = 1e-5
learning_rate0 = 1e-5
learning_rate1 = 1e-5
learning_rate2 = 1e-5


grad_all = 0.0
for t in range(10000):
   
    y_pred0 = model0(x)
    loss0 = loss_fn(y_pred0, y)


    y_pred1 = model1(x)
    loss1 = loss_fn(y_pred1, y)

    y_pred2 = model2(x)
    loss2 = loss_fn(y_pred2, y)


    #if t % 100 == 99:
    #    print(t, loss0.item(),'\t',loss1.item(),'\t', loss2.item())


    # Zero the gradients before running the backward pass.
    model0.zero_grad()
    model1.zero_grad()
    model2.zero_grad()
    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss0.backward()
    loss1.backward()
    loss2.backward()


    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients

    with torch.no_grad():
        for param0 in model0.parameters():
            param0 -= learning_rate0 * param0.grad

    with torch.no_grad():
        for param1 in model1.parameters():
            param1 -= learning_rate1 * param1.grad

    with torch.no_grad():
        for param2 in model2.parameters():
            param2 -= learning_rate2 * param2.grad
    
    if t % 100 == 99:
        print(t, loss0.item(),'\t',loss1.item(),'\t', loss2.item())
        loss_values0.append(loss0.item())
        loss_values1.append(loss1.item())
        loss_values2.append(loss2.item())
    



#print(loss_values0)
plt.plot(loss_values0,label='model0')
plt.plot(loss_values1,label='model1')
plt.plot(loss_values2,label='model2')

plt.title('Loss')
plt.xlabel('Epochs/*100')
plt.ylabel('Loss')
plt.legend()
plt.show()




plt.plot(x,y, label='Sin(x)/x')
plt.plot(x,model0(x).detach().numpy(), label='model0')
plt.plot(x,model1(x).detach().numpy(), label='model1')
plt.plot(x,model2(x).detach().numpy(), label='model2')


plt.title('Input (x) versus Output (y)')
plt.xlabel('Input Variable (x)')
plt.ylabel('Output Variable (y)')
plt.legend()
plt.show()


