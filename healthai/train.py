# train.py
import os
import torch, logging, torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from data_loader import get_loader
from models import MultiModalNet
from config import Config

# Ensure output directories exist
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)

def train():
    logging.basicConfig(level=logging.INFO)
    writer = SummaryWriter(Config.LOG_DIR)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiModalNet().to(device)
    opt   = AdamW(model.parameters(), lr=Config.LR)
    sched = CosineAnnealingLR(opt, Config.EPOCHS)
    scaler= GradScaler()
    crit  = nn.CrossEntropyLoss()

    loader = get_loader(f"{Config.DATA_DIR}/xray.csv",
                        'xray', Config.BATCH_SIZE, augment=True)
    for ep in range(1, Config.EPOCHS+1):
        model.train(); total=0
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            with autocast():
                preds = model({'xray': x})
                loss  = crit(preds, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            total += loss.item()
        sched.step()
        logging.info(f"Epoch {ep}/{Config.EPOCHS}, Loss={total/len(loader):.4f}")
        writer.add_scalar('Train/Loss', total/len(loader), ep)
        if ep % 10 == 0:
            torch.save(model.state_dict(),
                       f"{Config.MODEL_DIR}/model_ep{ep}.pth")

if __name__ == '__main__':
    train()
