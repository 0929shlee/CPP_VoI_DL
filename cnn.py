import torch


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=4, stride=1, padding=1),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = torch.nn.Linear(50688, 1, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def run(cpp_dataset):
    x_train, y_train = cpp_dataset.gen_dataset_cnn(type='train')
    x_test, y_test = cpp_dataset.gen_dataset_cnn(type='test')

    model = CNN()
    criterion = torch.nn.functional.mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print('*** Start learning')
    for epoch in range(50):
        optimizer.zero_grad()
        hypothesis = model(x_train)
        cost = criterion(hypothesis, y_train)
        cost.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print('Epoch: {} cost = {}'.format(epoch, cost.item()))

    print('Learning finished')

    with torch.no_grad():
        prediction = model(x_test)
        predict_aoi_1d = [pred.item() for pred in prediction]
        return predict_aoi_1d
