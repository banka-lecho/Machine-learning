import torch
from sklearn.metrics import classification_report


def train_on_batch(model, x_batch, y_batch, optimizer, loss_function, device):
    """Обучение на одном батче данных"""
    model.train()
    model.zero_grad()

    output = model(x_batch.to(device))

    loss = loss_function(output, y_batch.to(device))
    loss.backward()

    optimizer.step()
    return loss.cpu().item()


def train_epoch(train_generator, model, loss_function, optimizer, device, callback=None):
    epoch_loss = 0
    total = 0
    for it, (batch_of_x, batch_of_y) in enumerate(train_generator):
        batch_loss = train_on_batch(model, batch_of_x.to(device), batch_of_y.to(device), optimizer, loss_function)

        if callback is not None:
            callback(model, batch_loss)

        epoch_loss += batch_loss * len(batch_of_x)
        total += len(batch_of_x)

    return epoch_loss / total


def trainer(count_of_epoch,
            batch_size,
            dataset,
            model,
            loss_function,
            optimizer,
            device,
            lr=0.001,
            callback=None):
    train_test_loss = []
    optima = optimizer(model.parameters(), lr=lr)
    for _ in range(count_of_epoch):
        batch_generator = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        epoch_loss = train_epoch(train_generator=batch_generator,
                                 model=model,
                                 loss_function=loss_function,
                                 optimizer=optima,
                                 callback=callback,
                                 device=device)
        train_test_loss.append(epoch_loss)
    return model, train_test_loss


def tester(model, test_dataset, loss_function, device):
    batch_generator = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=64)
    pred = []
    real = []
    test_loss = 0
    for it, (x_batch, y_batch) in enumerate(batch_generator):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        output = model(x_batch)

        test_loss += loss_function(output, y_batch).cpu().item() * len(x_batch)

        pred.extend(torch.argmax(output, dim=-1).cpu().numpy().tolist())
        real.extend(y_batch.cpu().numpy().tolist())

    report = classification_report(real, pred, output_dict=True)
    return report['macro avg']['f1-score'], classification_report(real, pred)
