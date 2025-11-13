from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch
import os
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.dataloader import build_dataloader
from src.models.agent_model import AgentModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", device="cpu"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.to(device)

    def encode(self, texts):
        # texts: list[str]
        toks = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = toks["input_ids"].to(self.device)
        attn = toks["attention_mask"].to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attn)
        # pooled CLS embedding
        return out.pooler_output  # (B, hidden_dim)

def collate_fn(batch):
    # batch: list of dicts from EmbodiedDataset
    instrs = [b["instruction"] for b in batch]
    # demo_images: dataset provides demo_images (num_demos,3,H,W) per example
    # average demo frames per example to get one demo image (simple first step)
    demo_imgs = torch.stack([b["demo_images"].mean(dim=0) for b in batch], dim=0)
    current_imgs = torch.stack([b["current_image"] for b in batch], dim=0)
    # targets: take last valid action from first demo (or -1 padded)
    targets = []
    for b in batch:
        seq = b["demo_actions"][0]  # (max_steps,7)
        valid = (seq != -1).all(dim=1)
        idxs = valid.nonzero(as_tuple=False)
        if len(idxs) == 0:
            targets.append([-1]*7)
        else:
            last = idxs[-1].item()
            targets.append(seq[last].tolist())
    targets = torch.tensor(targets, dtype=torch.long)
    return instrs, demo_imgs, current_imgs, targets

def train(
    data_root=None,
    batch_size=8,
    epochs=3,
    lr=1e-4,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    train_dl = build_dataloader(batch_size=batch_size, debug=False, data_root=data_root, num_workers=2)
    val_dl = build_dataloader(batch_size=batch_size, debug=True, data_root=data_root, num_workers=2)

    # wrap dataloaders with collate
    train_dl = DataLoader(train_dl.dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_dl.dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # model
    bins = [101,101,101,121,121,121,2]
    model = AgentModel(token_dim=256, out_dim=512, bins=bins, device=device).to(device)

    # optimizer: only trainable params (projections + policy + heads)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)

    loss_fn = None  # OutputHeads.loss used below

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch} train")
        running_loss = 0.0
        for instrs, demo_imgs, current_imgs, targets in pbar:
            demo_imgs = demo_imgs.to(device)
            current_imgs = current_imgs.to(device)
            targets = targets.to(device)

            logits = model.forward(instrs, demo_imgs, current_imgs)
            loss = model.heads.loss(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (1 + pbar.n))

        # validation
        model.eval()
        correct = [0]*7
        total = 0
        with torch.no_grad():
            for instrs, demo_imgs, current_imgs, targets in tqdm(val_dl, desc="val"):
                demo_imgs = demo_imgs.to(device)
                current_imgs = current_imgs.to(device)
                targets = targets.to(device)
                logits = model.forward(instrs, demo_imgs, current_imgs)
                preds = model.heads.predict(logits).cpu()
                tgt = targets.cpu()
                mask = (tgt != -1)
                for i in range(7):
                    valid = mask[:, i]
                    if valid.sum().item() == 0:
                        continue
                    correct[i] += (preds[valid, i] == tgt[valid, i]).sum().item()
                total += tgt.size(0)
        accs = [c / max(1, total) for c in correct]
        print(f"Epoch {epoch} loss={running_loss/len(train_dl):.4f} val_accs={accs}")

        # save single checkpoint
        ckpt = {"model_state": model.state_dict(), "bins": bins}
        torch.save(ckpt, f"checkpoints/agent_epoch{epoch}.pt")

if __name__ == "__main__":
    # simple CLI-friendly entry
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()
    train(data_root=args.data_root, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)