import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Attempt to import AutoAttack
try:
    from autoattack import AutoAttack
    AUTOATTACK_AVAILABLE = True
except ImportError:
    AUTOATTACK_AVAILABLE = False
    print("Warning: AutoAttack not installed.")

from models.lips_ziphreg import LipsFormerSwin

# --- LOGGER ---
class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message); self.log.write(message); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()

# ------------------------------------------------------------------------------
# 1. Utilities
# ------------------------------------------------------------------------------
def save_attention_plots(model, loader, device, epoch, args, target_cdf):
    model.eval()
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    with torch.no_grad():
        try:
            data, _ = next(iter(loader))
        except StopIteration: return
        data = data.to(device)
        _, attn_weights = model(data, return_attn=True)
        if attn_weights is None: return
        attn = attn_weights[-1] if isinstance(attn_weights, list) else attn_weights
        B, H, Q, K = attn.shape
        flat_attn = attn.view(-1, K)
        sorted_attn, _ = torch.sort(flat_attn, dim=-1, descending=True)
        current_cdfs = torch.cumsum(sorted_attn, dim=-1)
        avg_cdf_curve = current_cdfs.mean(dim=0).cpu().numpy()
        avg_cdf_curve = avg_cdf_curve / (avg_cdf_curve[-1] + 1e-8)
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, 1, len(avg_cdf_curve)), avg_cdf_curve, label=f'Epoch {epoch}')
        if target_cdf is not None:
            t_np = target_cdf.view(-1).cpu().numpy()
            plt.plot(np.linspace(0, 1, len(t_np)), t_np, 'r--', label='Target')
        plt.savefig(os.path.join(plot_dir, f"cdf_epoch_{epoch}.png"))
        plt.close()

def compute_zipfian_loss(attn_weights, target_cdf):
    loss = 0.0
    target_base = target_cdf.view(-1)
    for attn in attn_weights:
        B, H, N, _ = attn.shape
        sorted_weights, _ = torch.sort(attn.view(-1, N), dim=-1, descending=True)
        current_cdf = torch.cumsum(sorted_weights, dim=-1)
        if target_base.shape[0] != N:
            temp_target = target_base.view(1, 1, -1)
            resized = torch.nn.functional.interpolate(temp_target, size=N, mode='linear', align_corners=False)
            resized_target = resized.view(-1)
            resized_target = resized_target / (resized_target.max() + 1e-8)
        else:
            resized_target = target_base
        target = resized_target.unsqueeze(0).expand_as(current_cdf)
        loss += torch.nn.functional.l1_loss(current_cdf, target)
    return loss / len(attn_weights)

def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, steps=10, device='cuda'):
    model.eval()
    adv = images.clone().detach() + torch.empty_like(images).uniform_(-eps, eps)
    adv = torch.clamp(adv, 0, 1).detach()
    for _ in range(steps):
        adv.requires_grad = True
        outputs = model(adv)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        delta = torch.clamp(adv - images, -eps, eps)
        adv = torch.clamp(images + delta, 0, 1).detach()
    return adv

def test(model, device, test_loader, pgd_steps=0, desc="Eval", limit_batches=None):
    model.eval()
    correct, correct_pgd, total = 0, 0, 0
    print(f"Running {desc}...", end=" ", flush=True)
    for batch_idx, (data, target) in enumerate(test_loader):
        if limit_batches is not None and batch_idx >= limit_batches: break
        data, target = data.to(device), target.to(device)
        total += target.size(0)
        with torch.no_grad():
            output = model(data)
            correct += output.argmax(dim=1, keepdim=True).eq(target.view_as(output.argmax(dim=1, keepdim=True))).sum().item()
        if pgd_steps > 0:
            adv_data = pgd_attack(model, data, target, steps=pgd_steps, device=device)
            with torch.no_grad():
                output_adv = model(adv_data)
                correct_pgd += output_adv.argmax(dim=1, keepdim=True).eq(target.view_as(output_adv.argmax(dim=1, keepdim=True))).sum().item()
    acc = 100. * correct / total
    acc_pgd = 100. * correct_pgd / total if pgd_steps > 0 else 0.0
    print(f"| Clean: {acc:.2f}%" + (f" | PGD-{pgd_steps}: {acc_pgd:.2f}%" if pgd_steps > 0 else ""))
    return acc, acc_pgd

def run_autoattack(model, test_loader, device, model_name="Model"):
    if not AUTOATTACK_AVAILABLE: return
    print(f"\n>>> AUTOATTACK: {model_name} <<<")
    model.eval()
    all_imgs, all_lbls = [], []
    for data, target in test_loader:
        all_imgs.append(data); all_lbls.append(target)
    x_test, y_test = torch.cat(all_imgs, dim=0), torch.cat(all_lbls, dim=0)
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard')
    with torch.no_grad(): adversary.run_standard_evaluation(x_test, y_test, bs=100)

# ------------------------------------------------------------------------------
# 2. Main
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='tiny_imgnet_final')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--reg-warmup-start', type=int, default=10)
    parser.add_argument('--reg-warmup-epochs', type=int, default=10)
    parser.add_argument('--decay-start', type=int, default=75)
    parser.add_argument('--lambda-reg', type=float, default=150.0)
    parser.add_argument('--lambda-min', type=float, default=15.0)
    parser.add_argument('--pulse-batches', type=int, default=40)
    parser.add_argument('--output-dir', type=str, default='logs_tiny_imagenet')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = TeeLogger(os.path.join(args.output_dir, f"{args.name}.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LipsFormerSwin(img_size=64, patch_size=4, in_chans=3, num_classes=200, 
                           embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], 
                           window_size=8, mlp_ratio=4.).to(device)

    norm = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
    t_train = transforms.Compose([transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    t_test = transforms.Compose([transforms.ToTensor(), norm])
    train_loader = DataLoader(datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=t_train), batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=t_test), batch_size=args.batch_size, shuffle=False, num_workers=4)

    win_sq = 8 * 8
    ranks = torch.arange(1, win_sq + 1, dtype=torch.float32, device=device)
    target_cdf = torch.cumsum((1.0 / (ranks ** 0.8)) / (1.0 / (ranks ** 0.8)).sum(), dim=0)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 1
    best_pgd_acc, best_path = 0.0, os.path.join(args.output_dir, f"{args.name}_best_pulse.pth")
    ckpt_path = os.path.join(args.output_dir, f"{args.name}_checkpoint.pth")

    # --- RESUME LOGIC ---
    if args.resume and os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_pgd_acc = checkpoint.get('best_pgd_acc', 0.0)

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch < args.reg_warmup_start: current_lambda = 0.0
        elif epoch < (args.reg_warmup_start + args.reg_warmup_epochs):
            current_lambda = args.lambda_reg * ((epoch - args.reg_warmup_start) / args.reg_warmup_epochs)
        elif epoch >= args.decay_start:
            prog = (epoch - args.decay_start) / (args.epochs - args.decay_start)
            current_lambda = max(args.lambda_reg - prog * (args.lambda_reg - args.lambda_min), args.lambda_min)
        else: current_lambda = args.lambda_reg

        model.train()
        print(f"\n>>> EPOCH {epoch} | Lambda: {current_lambda:.4f} <<<")
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, attn_weights = model(data, return_attn=True)
            loss = nn.CrossEntropyLoss()(output, target) + (current_lambda * compute_zipfian_loss(attn_weights, target_cdf))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # Save Checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_pgd_acc': best_pgd_acc
            }, ckpt_path)

        if epoch % 10 == 0 or (epoch >= args.decay_start and epoch <= args.decay_start + 40 and epoch % 2 == 0):
            save_attention_plots(model, val_loader, device, epoch, args, target_cdf)
            _, acc_pgd = test(model, device, val_loader, pgd_steps=10, desc="PULSE", limit_batches=args.pulse_batches)
            if acc_pgd > best_pgd_acc:
                best_pgd_acc = acc_pgd
                torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.name}_final.pth"))
    run_autoattack(model, val_loader, device, "Final")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        run_autoattack(model, val_loader, device, "Best Pulse")

if __name__ == '__main__': main()
