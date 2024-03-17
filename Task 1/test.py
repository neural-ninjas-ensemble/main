import torch
from torch.nn.functional import mse_loss


def eval(device, epoch, model, criterion, val_loader):
    model.eval()

    running_loss = 0.
    l2_running_loss = 0.
    num_batches = 0
    with torch.no_grad():
        for batch_idx, (id_, data, label, target_emb) in enumerate(val_loader):
            data = data.float().to(device)
            target_emb = target_emb.to(device)
            our_emb = model(data)

            loss = criterion(target_emb, our_emb)
            loss_l2 = mse_loss(our_emb, target_emb)

            running_loss += loss.item()
            l2_running_loss += loss_l2.item()
            num_batches += 1

    spec_loss = running_loss / num_batches
    l2_loss = l2_running_loss / num_batches
    print(f"Epoch: {epoch} | Loss: {spec_loss:.4f} | L2-loss: {l2_loss:.4f}")
    return spec_loss, l2_loss
