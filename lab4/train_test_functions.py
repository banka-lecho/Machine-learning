import torch
import numpy as np
from tqdm import tqdm


def train_epoch(train_generator, model, loss_function, optimizer, train_on_batch, device, callback=None):
    """Обучиться на одном батче"""
    epoch_loss = 0
    total = 0
    # итерируемся по батчам
    for it, (batch_of_x, batch_of_y) in enumerate(train_generator):
        batch_loss = train_on_batch(model, batch_of_x.to(device), batch_of_y.to(device), optimizer, loss_function,
                                    device)

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
            batch_func,
            device,
            lr=0.001,
            callback=None):

    optima = optimizer(model.parameters(), lr=lr)
    iterations = tqdm(range(count_of_epoch), desc='epoch')
    iterations.set_postfix({'train epoch loss': np.nan})
    for _ in iterations:
        # батч-генератор в формате торча, потому что датасет тоже в этом формате
        batch_generator = tqdm(
            torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True),
            leave=False, total=len(dataset) // batch_size + (len(dataset) % batch_size > 0))

        epoch_loss = train_epoch(train_generator=batch_generator,
                                 model=model,
                                 loss_function=loss_function,
                                 optimizer=optima,
                                 train_on_batch=batch_func,
                                 device=device,
                                 callback=callback)

        iterations.set_postfix({'train epoch loss': epoch_loss})


def test_model(model, device, test_dataset, loss_function):
    batch_generator = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=24)
    pred = []
    real = []
    test_loss = 0
    x_batch, y_batch = None, None
    for it, (x_batch, y_batch) in enumerate(batch_generator):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.no_grad():  # мб не надо
            output = model(x_batch)

        test_loss += loss_function(output, y_batch).cpu().item() * len(x_batch)

        pred.extend(torch.argmax(output, dim=-1).cpu().numpy().tolist())
        real.extend(y_batch.cpu().numpy().tolist())

    test_loss /= len(test_dataset)

    print('loss: {}'.format(test_loss))
    return x_batch, y_batch
