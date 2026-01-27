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
    print("Warning: AutoAttack not installed. Final robust evaluation will be skipped.")

from models.lips_ziphreg import LipsFormerSwin

# --- LOGGER CLASS (DUAL OUTPUT) ---
class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ------------------------------------------------------------------------------
# 1. Visualization & Plotting
# ------------------------------------------------------------------------------
def save_attention_plots(model, loader, device, epoch, args, target_cdf):
    model.eval()
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    with torch.no_grad():
        try:
            data, _ = next(iter(loader))
        except StopIteration:
            return
        data = data.to(device)
        
        try:
            _, attn_weights = model(data, return_attn=True)
        except TypeError:
            return 

        if attn_weights is None: return
        attn = attn_weights[-1] if isinstance(attn_weights, list) else attn_weights

        B, H, Q, K = attn.shape
        flat_attn = attn.view(-1, K)
        sorted_attn, _ = torch.sort(flat_attn, dim=-1, descending=True)
        current_cdfs = torch.cumsum(sorted_attn, dim=-1)
        
        avg_cdf_curve = current_cdfs.mean(dim=0).cpu().numpy()
        avg_cdf_curve = avg_cdf_curve / (avg_cdf_curve[-1] + 1e-8)

        n = len(avg_cdf_curve)
        x_axis = np.linspace(0, 1, n)
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, avg_cdf_curve, label=f'Epoch {epoch} Actual', linewidth=2.5)
        
        if target_cdf is not None:
             t_np = target_cdf.view(-1).cpu().numpy()
             if len(t_np) != n:
                 t_x_old = np.linspace(0, 1, len(t_np))
                 t_np = np.interp(x_axis, t_x_old, t_np)
             plt.plot(x_axis, t_np, 'r--', label='Target Zipf', linewidth=2)

        plt.title(f'Attention CDF (Epoch {epoch}) - {args.name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, f"{args.name}_cdf_epoch_{epoch}.png"))
        plt.close()
    model.train()

# ------------------------------------------------------------------------------
# 2. Robustness Utilities
# ------------------------------------------------------------------------------
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
    images, labels = images.to(device), labels.to(device)
    adv_images = images.clone().detach() + torch.empty_like(images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, 0, 1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, -eps, eps)
        adv_images = torch.clamp(images + delta, 0, 1).detach()
    return adv_images

def test(model, device, test_loader, pgd_steps=0, desc="Eval", limit_batches=None):
    model.eval()
    correct, correct_pgd, total = 0, 0, 0
    print(f"Running {desc}...", end=" ", flush=True)
    
    for batch_idx, (data, target) in enumerate(test_loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        data, target = data.to(device), target.to(device)
        total += target.size(0)
        
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        if pgd_steps > 0:
            adv_data = pgd_attack(model, data, target, steps=pgd_steps, device=device)
            with torch.no_grad():
                output_adv = model(adv_data)
                pred_adv = output_adv.argmax(dim=1, keepdim=True)
                correct_pgd += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    acc = 100. * correct / total
    acc_pgd = 100. * correct_pgd / total if pgd_steps > 0 else 0.0
    print(f"| Clean: {acc:.2f}%" + (f" | PGD-{pgd_steps}: {acc_pgd:.2f}%" if pgd_steps > 0 else ""))
    return acc, acc_pgd

def run_autoattack(model, test_loader, device, model_name="Model"):
    if not AUTOATTACK_AVAILABLE: return
    print(f"\n{'='*60}\n>>> STARTING AUTOATTACK: {model_name} (Linf, 8/255) <<<\n{'='*60}")
    model.eval()
    all_imgs, all_lbls = [], []
    for data, target in test_loader:
        all_imgs.append(data); all_lbls.append(target)
    x_test, y_test = torch.cat(all_imgs, dim=0), torch.cat(all_lbls, dim=0)
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard')
    with torch.no_grad():
        adversary.run_standard_evaluation(x_test, y_test, bs=100)

# ------------------------------------------------------------------------------
# 3. Training Loop
# ------------------------------------------------------------------------------
def train_epoch(args, model, device, train_loader, optimizer, epoch, target_cdf, current_lambda):
    print(f"\n>>> EPOCH {epoch} | Lambda: {current_lambda:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} <<<")
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, attn_weights = model(data, return_attn=True)
        
        loss_ce = criterion(output, target)
        raw_reg = compute_zipfian_loss(attn_weights, target_cdf) if current_lambda > 0 else torch.tensor(0.0).to(device)
        
        loss = loss_ce + (current_lambda * raw_reg)
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print(f"  Batch {batch_idx:04d}: CE {loss_ce.item():.4f} | Reg Raw {raw_reg.item():.5f} | Total {loss.item():.4f}")

# ------------------------------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='tiny_imgnet_dual_aa')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to Tiny-ImageNet-200 root')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--reg-warmup-start', type=int, default=10)
    parser.add_argument('--reg-warmup-epochs', type=int, default=10)
    parser.add_argument('--decay-start', type=int, default=100)
    parser.add_argument('--lambda-reg', type=float, default=200.0)
    parser.add_argument('--lambda-min', type=float, default=20.0)
    parser.add_argument('--pulse-batches', type=int, default=40)
    parser.add_argument('--output-dir', type=str, default='logs_tiny_imagenet')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"{args.name}.log")
    sys.stdout = TeeLogger(log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"======================================================\n STARTING: {args.name}\n======================================================")

    # 1. Swin-Tiny Configuration
    model = LipsFormerSwin(
        img_size=64, patch_size=4, in_chans=3, num_classes=200,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=8, mlp_ratio=4.
    ).to(device)

    # 2. Data Loaders
    norm = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
    t_train = transforms.Compose([transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    t_test = transforms.Compose([transforms.ToTensor(), norm])
    
    train_loader = DataLoader(datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=t_train), batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=t_test), batch_size=128, shuffle=False, num_workers=4)

    # 3. Target Distribution (0.8 Exponent)
    win_sq = 8 * 8
    ranks = torch.arange(1, win_sq + 1, dtype=torch.float32, device=device)
    target_pdf = 1.0 / (ranks ** 0.8)
    target_cdf = torch.cumsum(target_pdf / target_pdf.sum(), dim=0)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_pgd_acc = 0.0
    best_model_path = os.path.join(args.output_dir, f"{args.name}_best_pulse.pth")
    final_model_path = os.path.join(args.output_dir, f"{args.name}_final.pth")

    for epoch in range(1, args.epochs + 1):
        # --- Lambda Scheduler Logic ---
        if epoch < args.reg_warmup_start:
            current_lambda = 0.0
        elif epoch < (args.reg_warmup_start + args.reg_warmup_epochs):
            current_lambda = args.lambda_reg * ((epoch - args.reg_warmup_start) / args.reg_warmup_epochs)
        elif epoch >= args.decay_start:
            prog = (epoch - args.decay_start) / (args.epochs - args.decay_start)
            current_lambda = max(args.lambda_reg - prog * (args.lambda_reg - args.lambda_min), args.lambda_min)
        else:
            current_lambda = args.lambda_reg

        train_epoch(args, model, device, train_loader, optimizer, epoch, target_cdf, current_lambda)
        scheduler.step()

        # --- Pulse Check Evaluation & Plots ---
        is_crit = (epoch >= args.decay_start and epoch <= args.decay_start + 40)
        if epoch % 10 == 0 or (is_crit and epoch % 2 == 0) or epoch == args.epochs:
            save_attention_plots(model, val_loader, device, epoch, args, target_cdf)
            _, acc_pgd = test(model, device, val_loader, pgd_steps=10, desc="PULSE CHECK", limit_batches=args.pulse_batches)
            
            if acc_pgd > best_pgd_acc:
                best_pgd_acc = acc_pgd
                print(f"  --> New Best Robustness: {best_pgd_acc:.2f}% | Saving Checkpoint...")
                torch.save(model.state_dict(), best_model_path)

    # --- FINAL EVALUATION ---
    print("\n\n>>> TRAINING COMPLETE. SAVING FINAL MODEL. <<<")
    torch.save(model.state_dict(), final_model_path)
    
    # 1. Run AutoAttack on Final Model
    run_autoattack(model, val_loader, device, model_name="Final Model")
    
    # 2. Run AutoAttack on Best Pulse Model
    if os.path.exists(best_model_path):
        print("\n>>> LOADING BEST PULSE MODEL FOR EVALUATION... <<<")
        model.load_state_dict(torch.load(best_model_path))
        run_autoattack(model, val_loader, device, model_name="Best Pulse Model")

if __name__ == '__main__':
    main()
