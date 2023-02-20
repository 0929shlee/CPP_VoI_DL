import torch
import torch.nn as nn


def run(cpp_dataset, n_epochs):
    x_train, y_train = cpp_dataset.gen_dataset_dnn(type='train')
    x_test, y_test = cpp_dataset.gen_dataset_dnn(type='test')

    model = nn.Sequential(
        nn.Linear(1 * 101, 128, bias=True),
        nn.Sigmoid(),
        nn.Linear(128, 256, bias=True),
        nn.Sigmoid(),
        nn.Linear(256, 1, bias=True),
        nn.Sigmoid(),
    )

    criterion = torch.nn.functional.mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    predict_aoi_2d = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        hypothesis = model(x_train)
        cost = criterion(hypothesis, y_train)
        cost.backward()
        optimizer.step()

        with torch.no_grad():
            prediction = model(x_test)
            predict_aoi_1d = [pred.item() for pred in prediction]
            predict_aoi_2d.append(predict_aoi_1d)

        print('\rEpoch: {}/{} cost = {:.3f}'.format(epoch, n_epochs, cost.item()), end="")

    return predict_aoi_2d
