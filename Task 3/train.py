def train_epoch(device, model, criterion, optimizer, train_loader):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()

        data = data.float().to(device)

        output = model(data)
        loss = criterion(label, output)

        loss.backward()
        optimizer.step()