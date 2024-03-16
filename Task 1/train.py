
def train_epoch(device, model, criterion, optimizer, train_loader):
    model.train()
    for batch_idx, (id_, data, label, target_emb) in enumerate(train_loader):
        optimizer.zero_grad()

        data = data.float().to(device)

        our_emb = model(data)
        loss = criterion(target_emb, our_emb)

        loss.backward()
        optimizer.step()
