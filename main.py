import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim


# MNIST
def get_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist_data', train=True, download = True,
                                                              transform = transformimport torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim


# MNIST
def get_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist_data', train=True, download = True,
                                                              transform = transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.1307,),(0.3081,))
                                                ])), batch_size = params['batch_size'], shuffle = True)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist_data', train=False, download = True,
                                                              transform = transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.1307,),(0.3081,))
                                                ])), batch_size = params['batch_size'], shuffle = True)
    return train_loader, test_loader                                                



'''
# CIFAR10
def get_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./cifar10_data', train=True, download = True,
                                                              transform = transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  # R, G, B
                                                ])), batch_size = 16, shuffle = True)

    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./cifar10_data', train=False, download = True,
                                                              transform = transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                                ])), batch_size = 16, shuffle = True)
    return train_loader, test_loader
'''

class XNORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.BinConv2d_1 = BinConv2d(1, 6,  kernel_size=5)
        self.BinConv2d_2 = BinConv2d(6, 16, kernel_size=3)
        self.fc1         = BinLinear(400, 50)
        self.fc2         = BinLinear(50, 10)

    def forward(self, I):
        I = self.BinConv2d_1(I)
        I = self.BinConv2d_2(I)
        I = I.view(-1, 400)
        I = self.fc1(I)
        I = F.relu(I)
        I = self.fc2(I)
        return I


class BinConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.bn           = nn.BatchNorm2d(in_channels) # default eps = 1e-5, momentum = 0.1, affine = True
        self.conv         = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size)
        self.relu         = nn.ReLU()
        self.pool         = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, I):
        I = self.bn(I)
        A = BinActiv().Mean(I)
        I = BinActive(I)
        k = torch.ones(1,1,self.kernel_size,self.kernel_size).mul(1/(self.kernel_size**2)) # 4d - batch,channel,height,width
        K = F.conv2d(A,k) # default stride=1, padding=0
        I = self.conv(I)
        I = torch.mul(I, K)
        I = self.relu(I)
        I = self.pool(I)

        return I



class BinActiv(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.sign(input)

        return input

    def Mean(self, input):
        return torch.mean(input.abs(), 1, keepdim=True)  # 1: channel // batch[0], channel[1], height[2], width[3]


    @staticmethod
    def backward(ctx, grad_output):
        input,  = ctx.saved_tensors

        # STE (Straight Through Estimator)
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0    # ge: greater or equal
        grad_input[input.le(-1)] = 0   # le: less or equal
        return grad_input

BinActive = BinActiv.apply

class BinLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_feature  = in_features
        self.out_feature = out_features
        self.bn          = nn.BatchNorm1d(in_features)
        self.linear      = nn.Linear(in_features, out_features)

    def forward(self, I):
        I = self.bn(I)
        beta = BinActiv().Mean(I).expand_as(I)
        I = BinActive(I)
        I = torch.mul(I, beta)
        I = self.linear(I)
        return I



class WeightOperation:
    def __init__(self,model):

        self.count_group_weights = 0
        self.weight = []
        self.saved_weight = []


        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):

                self.count_group_weights += 1
                self.weight.append(m.weight)
                self.saved_weight.append(m.weight.data)


    def WeightSave(self):
        for index in range(self.count_group_weights):
            self.saved_weight[index].copy_(self.weight[index].data)


    def WeightBinarize(self):
        for index in range(self.count_group_weights):

            n                 = self.weight[index].data[0].nelement()
            dim_group_weights = self.weight[index].data.size()

            if len(dim_group_weights) == 4:
                alpha = self.weight[index].data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(dim_group_weights)

            elif len(dim_group_weights) == 2:
                alpha = self.weight[index].data.norm(1, 1, keepdim=True).div(n).expand(dim_group_weights)

            self.weight[index].data = self.weight[index].data.sign()* alpha


    def WeightRestore(self):
        for index in range(self.count_group_weights):
            self.weight[index].data.copy_(self.saved_weight[index])


    def WeightGradient(self):
        for index in range(self.count_group_weights):
            n = self.weight[index].data[0].nelement()
            dim_group_weights = self.weight[index].data.size()

            if len(dim_group_weights) == 4:
                alpha = self.weight[index].data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(dim_group_weights)

            elif len(dim_group_weights) == 2:
                alpha = self.weight[index].data.norm(1, 1, keepdim=True).div(n).expand(dim_group_weights)

            alpha[self.weight[index].data.le(-1.0)] = 0
            alpha[self.weight[index].data.ge( 1.0)] = 0

            self.weight[index].grad = self.weight[index].grad / n + self.weight[index].grad * alpha



model = XNORModel()
WeightOperation = WeightOperation(model)

optimizer = optim.Adam(model.parameters())
params = {'epochs':100, 'batch_size':32}
loss_fn   = torch.nn.CrossEntropyLoss()

train_loader, test_loader = get_loaders(batch_size=params['batch_size'])

for epoch in range(params['epochs']):

    # training
    for batch_idx, (train_inputs, train_labels) in enumerate(train_loader): # train_inputs size:[32,1,28,28], labels size: [32]
        optimizer.zero_grad()

        WeightOperation.WeightSave()
        WeightOperation.WeightBinarize()

        predicted = model(train_inputs)
        loss = loss_fn(predicted, train_labels)
        loss.backward()     # gradient

        WeightOperation.WeightRestore()
        WeightOperation.WeightGradient()

        optimizer.step()    # update


        if batch_idx % 100 ==0:
            print('[%d, %5d] loss: %.3f' %(epoch, batch_idx*len(train_inputs), loss.item()))  # loss: loss tensor(2.3027, grad_fn=<NllLossBackward>)


    # test
    correct = 0
    WeightOperation.WeightSave()
    WeightOperation.WeightBinarize()

    for (test_inputs, test_labels) in test_loader:
        predicted = model(test_inputs)
        pred = predicted.data.max(1, keepdim = False)[1] # max(0):column-wise, max(1):row-wise, [0]:values [1]:index
        correct += pred.eq(test_labels.data).sum()

        acc = 100. * correct / len(test_loader.dataset)

    WeightOperation.WeightRestore()

    print('Accuracy:', acc.item())


s.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.1307,),(0.3081,))
                                                ])), batch_size = params['batch_size'], shuffle = True)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist_data', train=False, download = True,
                                                              transform = transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.1307,),(0.3081,))
                                                ])), batch_size = params['batch_size'], shuffle = True)
    return train_loader, test_loader                                                



'''
# CIFAR10
def get_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./cifar10_data', train=True, download = True,
                                                              transform = transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  # R, G, B
                                                ])), batch_size = 16, shuffle = True)

    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./cifar10_data', train=False, download = True,
                                                              transform = transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                                ])), batch_size = 16, shuffle = True)
    return train_loader, test_loader
'''

class XNORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.BinConv2d_1 = BinConv2d(1, 6,  kernel_size=5)
        self.BinConv2d_2 = BinConv2d(6, 16, kernel_size=3)
        self.fc1         = BinLinear(400, 50)
        self.fc2         = BinLinear(50, 10)

    def forward(self, I):
        I = self.BinConv2d_1(I)
        I = self.BinConv2d_2(I)
        I = I.view(-1, 400)
        I = self.fc1(I)
        I = F.relu(I)
        I = self.fc2(I)
        return I


class BinConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.bn           = nn.BatchNorm2d(in_channels) # default eps = 1e-5, momentum = 0.1, affine = True
        self.conv         = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size)
        self.relu         = nn.ReLU()
        self.pool         = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, I):
        I = self.bn(I)
        A = BinActiv().Mean(I)
        I = BinActive(I)
        k = torch.ones(1,1,self.kernel_size,self.kernel_size).mul(1/(self.kernel_size**2)) # 4d - batch,channel,height,width
        K = F.conv2d(A,k) # default stride=1, padding=0
        I = self.conv(I)
        I = torch.mul(I, K)
        I = self.relu(I)
        I = self.pool(I)

        return I



class BinActiv(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.sign(input)

        return input

    def Mean(self, input):
        return torch.mean(input.abs(), 1, keepdim=True)  # 1: channel // batch[0], channel[1], height[2], width[3]

    @staticmethod
    def backward(ctx, grad_output):
        input,  = ctx.saved_tensors

        # STE (Straight Through Estimator)
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0    # ge: greater or equal
        grad_input[input.le(-1)] = 0   # le: less or equal
        return grad_input

BinActive = BinActiv.apply

class BinLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_feature  = in_features
        self.out_feature = out_features
        self.bn          = nn.BatchNorm1d(in_features)
        self.linear      = nn.Linear(in_features, out_features)

    def forward(self, I):
        I = self.bn(I)
        beta = BinActiv().Mean(I).expand_as(I)
        I = BinActive(I)
        I = torch.mul(I, beta)
        I = self.linear(I)
        return I



class BinOperation:
    def __init__(self,model):
        count_targets = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                count_targets += 1

        self.num_of_modules = len(np.linspace(0,count_targets-1,count_targets).astype('int').tolist())
        self.weight = []

        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):

                self.weight.append(m.weight) #Parameter


    def BinarizeWeights(self):
        for index in range(self.num_of_modules):
            print('self.weight[index].data:',self.weight[index].data)
            print('self.weight[index].data[0]:',self.weight[index].data[0])
            exit(-1)
            n = self.weight[index].data[0].nelement()
            dim_group_of_weights = self.weight[index].data.size()

            if len(dim_group_of_weights) == 4:
                alpha = self.weight[index].data.norm(1,3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True).div(n)

            elif len(dim_group_of_weights) == 2:
                alpha = self.weight[index].data.norm(1,1,keepdim=True).div(n)

            self.weight[index].data = self.weight[index].data.sign().mul(alpha.expand(dim_group_of_weights))


model = XNORModel()
optimizer = optim.Adam(model.parameters())
params = {'epochs':100, 'batch_size':32}
loss_fn   = torch.nn.CrossEntropyLoss()

train_loader, test_loader = get_loaders(batch_size=params['batch_size'])

for epoch in range(params['epochs']):

    for batch_idx, (train_inputs, train_labels) in enumerate(train_loader): # train_inputs size:[32,1,28,28], labels size: [32]
        optimizer.zero_grad()
        predicted = model(train_inputs)
        loss = loss_fn(predicted, train_labels)
        loss.backward()     # gradient
        optimizer.step()    # update


        if batch_idx % 100 ==0:
            print('[%d, %5d] loss: %.3f' %(epoch, batch_idx*len(train_inputs), loss.item()))  # loss: loss tensor(2.3027, grad_fn=<NllLossBackward>)

    correct = 0
    for (test_inputs, test_labels) in test_loader:
        predicted = model(test_inputs)
        pred = predicted.data.max(1,keepdim=True)[1]             # max(0):column-wise, max(1):row-wise, [0]:values [1]:index
        correct += pred.eq(test_labels.data.view_as(pred)).sum()
        acc = 100. * correct / len(test_loader.dataset)

    print('Accuracy:', acc.item())


