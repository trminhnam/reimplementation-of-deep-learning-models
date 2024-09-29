import torch


def get_batch(split, batch_size, block_size):
    # Sample random starting points for each batch
    start_ids = torch.randint(0, split.shape[0] - block_size - 1, (batch_size,))

    x = [split[i : i + block_size] for i in start_ids]
    y = [split[i + 1 : i + block_size + 1] for i in start_ids]
    x = torch.stack(x)
    y = torch.stack(y)
    return x, y


@torch.no_grad()
def estimate_loss(model, split, batch_size, block_size, steps, device):
    model.eval()
    total_losses = []
    for _ in range(steps):
        x, y = get_batch(split, batch_size, block_size)
        x, y = x.to(device), y.to(device)
        loss, _ = model(x, y)
        total_losses.append(loss.detach().cpu().item())
    model.train()
    return sum(total_losses) / len(total_losses)
