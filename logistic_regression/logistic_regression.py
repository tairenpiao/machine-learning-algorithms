import torch
import torch.nn as nn
import argparse
import torchvision
import tqdm




class Logistic_regression(nn.Module):
    def __init__(self):
        super(Logistic_regression, self).__init__()
        self.logistic = nn.Linear(784, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.logistic(x)
        x = self.sigmoid(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1,784) # transform 28 x 28 to 784
        # print(data.shape)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))



def eval(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm.tqdm(test_loader):

            data = data.view(-1,784)
            data, target = data.to(device), target.to(device)

            output = model(data) #100s

            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    



def main():
    parser = argparse.ArgumentParser(description="Logistic regression parser")
    parser.add_argument('--batch_size', type=int, default=100, metavar='N', help="Input the batch size") # metavar is the hint
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--lr', default=0.01)
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()

    device = torch.device("cpu")

    # train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.Compose))
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307, ), (0.3081, ))])
        )
    eval_dataset = torchvision.datasets.MNIST(
        root='./data', 
        download=True,
        train=False,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307, ), (0.3081, ))])
        )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
    # print(train_loader.item())

    logistic_regression_model = Logistic_regression().to(device)

    optimizer = torch.optim.SGD(logistic_regression_model.parameters(), lr=args.lr)

    for epoch in range(1, args.epoch):
        train(args, logistic_regression_model, device, train_loader, optimizer, epoch)
        eval(args, logistic_regression_model, device, eval_loader)



if __name__ == '__main__':
    main()
