# simple training loop for any model + dataloader
# model.forward() must return (logits, loss) when labels are provided
# or just return loss directly — both conventions work

import time
import os
import torch
import matplotlib.pyplot as plt

from utils.training.saves import cleanup_checkpoints, load_latest_checkpoint, save_checkpoint


def plot_loss_history(loss_history, save_dir):
    # loss_history: [(batch_idx, loss), ...]
    if not loss_history:
        return

    batches = [b for b, _ in loss_history]
    losses = [l for _, l in loss_history]

    plt.figure(figsize=(10, 6))
    plt.plot(batches, losses, marker='o', markersize=3)
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.title("training loss")
    plt.savefig(os.path.join(save_dir, "loss.png"), dpi=150, bbox_inches='tight')
    plt.close()


def fmt_time(s):
    d, h, m, sec = int(s // 86400), int((s % 86400) // 3600), int((s % 3600) // 60), int(s % 60)
    p = []
    if d: p.append(f"{d}d")
    if h: p.append(f"{h}h")
    if m: p.append(f"{m}m")
    p.append(f"{sec}s")
    return " ".join(p)


def simple_train(
    model,
    dataloader,
    optimizer,
    scheduler=None,
    max_batches=10000,
    accumulation_steps=1,
    clip_grad_norm=1.0,
    save_folder_path="weights",
    batches_per_log=10,
    batches_per_save=500,
):
    # returns avg loss over entire run
    assert len(dataloader) > 0
    minibatch_size = dataloader.batch_size

    model.cuda().bfloat16()
    os.makedirs(save_folder_path, exist_ok=True)

    meta = load_latest_checkpoint(save_folder_path, model, optimizer, scheduler)
    batch_idx = meta.get('batch', 0) if meta else 0
    # [(batch_idx, loss), ...]
    loss_history = meta.get('loss_history', []) if meta else []
    prev_train_time = meta.get('total_train_time', 0.0) if meta else 0.0

    session_start_time = time.time()
    total_loss = 0.0
    # how many batches we've done this call (for computing avg)
    batches_this_run = 0
    data_iter = iter(dataloader)

    model.train()

    while batch_idx < max_batches:
        optimizer.zero_grad()
        acc_loss = 0.0

        for _ in range(accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = batch.cuda()
            out = model(batch, labels=batch)
            # model returns (logits, loss) or just loss
            loss = out[1] if isinstance(out, tuple) else out
            (loss / accumulation_steps).backward()
            acc_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()

        batch_idx += 1
        batches_this_run += 1
        avg_loss = acc_loss / accumulation_steps
        total_loss += avg_loss

        if batch_idx % batches_per_log == 0:
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - session_start_time
            total_time = prev_train_time + elapsed
            print(f"Time: {fmt_time(total_time)} | Batch {batch_idx}/{max_batches} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
            loss_history.append((batch_idx, avg_loss))

        if batch_idx % batches_per_save == 0:
            path = f"{save_folder_path}/batch_{batch_idx}.pt"
            total_time = prev_train_time + (time.time() - session_start_time)
            meta = {'batch': batch_idx, 'loss_history': loss_history, 'total_train_time': total_time}
            save_checkpoint(path, model, optimizer, meta, scheduler)
            meta_dir = os.path.join(save_folder_path, "metadata")
            plot_loss_history(loss_history, meta_dir)
            print(f"Saved: {path}")
            cleanup_checkpoints(save_folder_path)

    # final save
    path = f"{save_folder_path}/batch_{batch_idx}.pt"
    total_time = prev_train_time + (time.time() - session_start_time)
    meta = {'batch': batch_idx, 'loss_history': loss_history, 'total_train_time': total_time}
    save_checkpoint(path, model, optimizer, meta, scheduler)
    meta_dir = os.path.join(save_folder_path, "metadata")
    plot_loss_history(loss_history, meta_dir)
    cleanup_checkpoints(save_folder_path)

    return total_loss / batches_this_run


def simple_eval(model, dataloader):
    # returns avg loss across entire dataloader
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.cuda()
            out = model(batch, labels=batch)
            loss = out[1] if isinstance(out, tuple) else out
            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / num_batches
