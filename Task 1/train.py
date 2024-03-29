
def train_epoch(device, model, criterion, optimizer, train_loader):
    model.train()
    for batch_idx, (id_, data, label, target_emb) in enumerate(train_loader):
        optimizer.zero_grad()

        data = data.float().to(device)
        target_emb = target_emb.to(device)

        our_emb = model(data)
        loss = criterion(target_emb, our_emb)

        loss.backward()
        optimizer.step()


def train_epoch_pretr(device, model, fc, criterion, optimizer, train_loader):
    model.train()
    for batch_idx, (id_, data, label, target_emb) in enumerate(train_loader):
        optimizer.zero_grad()

        data = data.float().to(device)
        label = label.to(device)

        our_emb = model(data)
        our_emb = fc(our_emb)
        loss = criterion(our_emb, label)

        loss.backward()
        optimizer.step()
