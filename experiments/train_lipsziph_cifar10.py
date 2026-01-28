import argparse
import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import csv

# Attempt to import AutoAttack (ensure 'pip install autoattack' is run)
try:
    from autoattack import AutoAttack
    AUTOATTACK_AVAILABLE = True
except ImportError:
    AUTOATTACK_AVAILABLE = False
    print("Warning: AutoAttack not installed. Final robust evaluation will be skipped.")

# Import the model (assuming it exists in the same directory structure)
from models.lips_ziphreg import LipsFormerSwin

# --- LOGGER CLASS ---
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
# 1. Plotting Utility (Fixed 45-degree bug per instruction)
# ------------------------------------------------------------------------------
def save_attention_plots(model, loader, device, epoch, args, target_cdf):
    model.eval()
    base_dir = getattr(args, 'save_dir', getattr(args, 'output_dir', 'logs_experiment'))
    plot_dir = os.path.join(base_dir, "plots")
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

        if isinstance(attn_weights, list): attn = attn_weights[-1] 
        else: attn = attn_weights

        # Flatten [B, H, Q, K] -> [Total_Tokens, K]
        B, H, Q, K = attn.shape
        flat_attn = attn.view(-1, K)
        
        # Sort & CDF per token
        sorted_attn, _ = torch.sort(flat_attn, dim=-1, descending=True)
        current_cdfs = torch.cumsum(sorted_attn, dim=-1)
        
        # Average
        avg_cdf_curve = current_cdfs.mean(dim=0).cpu().numpy()
        avg_cdf_curve = avg_cdf_curve / (avg_cdf_curve[-1] + 1e-8)

        # Plot
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
# 2. Loss & Helper Functions
# ------------------------------------------------------------------------------
def compute_zipfian_loss(attn_weights, target_cdf, order=1):
    loss = 0.0
    if target_cdf.dim() > 1:
        target_base = target_cdf.view(-1)
    else:
        target_base = target_cdf

    for attn in attn_weights:
        B, H, N, _ = attn.shape
        sorted_weights, _ = torch.sort(attn.view(-1, N), dim=-1, descending=True)
        current_cdf = torch.cumsum(sorted_weights, dim=-1)
        
        target_len = target_base.shape[0]
        if target_len != N:
            temp_target = target_base.view(1, 1, -1)
            resized = torch.nn.functional.interpolate(temp_target, size=N, mode='linear', align_corners=False)
            resized_target = resized.view(-1)
            resized_target = resized_target / (resized_target.max() + 1e-8)
        else:
            resized_target = target_base

        target = resized_target.unsqueeze(0).expand_as(current_cdf)
        loss += torch.nn.functional.l1_loss(current_cdf, target)

    return loss / len(attn_weights)

def lipschitz_margin_loss(logits, targets, margin=0.3):
    correct_scores = logits.gather(1, targets.view(-1, 1)).squeeze()
    logits_masked = logits.clone()
    logits_masked[torch.arange(logits.size(0)), targets] = -float('inf')
    runner_up_scores = logits_masked.max(dim=1)[0]
    return torch.relu(margin - (correct_scores - runner_up_scores)).mean()

def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, steps=10, device='cuda'):
    images = images.to(device)
    labels = labels.to(device)
    loss_fn = nn.CrossEntropyLoss()
    delta = torch.empty_like(images).uniform_(-eps, eps)
    adv_images = torch.clamp(images + delta, 0, 1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = loss_fn(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, -eps, eps)
        adv_images = torch.clamp(images + delta, 0, 1).detach()

    return adv_images

def train(args, model, device, train_loader, optimizer, epoch, criterion, target_cdf, target_ratio=0.1):
    print(f"\n>>> EPOCH {epoch} | Dynamic Balancing (Target: {target_ratio*100}%) <<<")
    model.train()
    
    train_loss = 0.0
    reg_loss_track = 0.0
    # Start with the user-provided lambda as a baseline
    current_lambda = args.lambda_reg 
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # 1. Forward Pass
        output, attn_weights = model(data, return_attn=True)
        
        # 2. Compute Task Loss (Main)
        loss_main = criterion(output, target)
        if args.use_margin:
            loss_main += lipschitz_margin_loss(output, target, margin=0.3)

        # 3. Compute Regularization Loss (Zipf)
        raw_reg = compute_zipfian_loss(attn_weights, target_cdf, order=args.reg_order)

        # 4. Dynamic Lambda Calibration (Every 50 batches to save compute)
        if batch_idx % 50 == 0:
            # Get grads for Task Loss only
            if args.target_ratio == 0.0:
                 current_lambda = 0.0
                 norm_main = 0.0 
                 norm_reg = 0.0
            else:
                grad_main = torch.autograd.grad(loss_main, model.parameters(), retain_graph=True, allow_unused=True)
                norm_main = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grad_main if g is not None]), 2)
    
                # Get grads for Reg Loss only
                grad_reg = torch.autograd.grad(raw_reg, model.parameters(), retain_graph=True, allow_unused=True)
                norm_reg = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grad_reg if g is not None]), 2)
    
                # Update lambda: current_lambda * norm_reg should = norm_main * target_ratio
                if norm_reg > 1e-8:
                    ideal_lambda = (norm_main * target_ratio) / norm_reg
                    # Use a momentum-style update to prevent lambda spikes
                current_lambda = 0.9 * current_lambda + 0.1 * ideal_lambda.item()

        # 5. Final Combined Loss
        total_loss = loss_main + (current_lambda * raw_reg)
        total_loss.backward()
        optimizer.step()

        # Tracking
        train_loss += total_loss.item()
        reg_loss_track += raw_reg.item()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Lambda {current_lambda:.4f} | Task Norm: {norm_main:.4f} | Reg Norm: {norm_reg:.4f}")

    print(f"Train End: Avg Loss: {train_loss/len(train_loader):.4f} | Final Lambda: {current_lambda:.4f}")

def test(model, device, test_loader, criterion, pgd_steps=0, desc="Eval", limit_batches=None):
    model.eval()
    test_loss = 0
    correct = 0
    correct_pgd = 0
    total = 0
    
    print(f"Running {desc}...", end=" ", flush=True)
    
    for batch_idx, (data, target) in enumerate(test_loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
            
        data, target = data.to(device), target.to(device)
        current_batch_size = target.size(0)
        total += current_batch_size
        
        with torch.no_grad():
            output = model(data)
            test_loss += criterion(output, target).item() * current_batch_size
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        if pgd_steps > 0:
            adv_data = pgd_attack(model, data, target, eps=8/255, alpha=2/255, steps=pgd_steps, device=device)
            with torch.no_grad():
                output_adv = model(adv_data)
                pred_adv = output_adv.argmax(dim=1, keepdim=True)
                correct_pgd += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    acc = 100. * correct / total
    if pgd_steps > 0:
        acc_pgd = 100. * correct_pgd / total
        print(f"| Clean: {acc:.2f}% | PGD-{pgd_steps}: {acc_pgd:.2f}% ({total} images)")
        return acc, acc_pgd
    else:
        print(f"| Clean: {acc:.2f}% ({total} images)")
        return acc, 0.0

def run_autoattack(model, test_loader, device, log_file):
    if not AUTOATTACK_AVAILABLE:
        print("AutoAttack not installed, skipping.")
        return

    print("\n>>> RUNNING AUTOATTACK (Standard: APGD-CE, APGD-T, FAB-T, Square) <<<")
    model.eval()
    
    # Collect all test data
    all_imgs = []
    all_lbls = []
    for data, target in test_loader:
        all_imgs.append(data)
        all_lbls.append(target)
    
    x_test = torch.cat(all_imgs, dim=0)
    y_test = torch.cat(all_lbls, dim=0)
    
    # Run AA (batch size 100 to avoid OOM)
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard')
    
    # AA writes to stdout, we also want to capture it in our log if possible
    # Note: AA usually handles its own logging, but we will print the result at the end
    with torch.no_grad():
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=100)
    
    # Calculate final accuracy
    # (AutoAttack prints detailed logs to stdout automatically)
    print("AutoAttack evaluation complete.")

# ------------------------------------------------------------------------------
# 3. Main Execution
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='experiment')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='logs_experiment')
    parser.add_argument('--lambda-reg', type=float, default=1.0) # Initial baseline
    parser.add_argument('--reg-order', type=int, default=1)         
    parser.add_argument('--use-margin', action='store_true')        
    parser.add_argument('--use-noise', action='store_true') 

    # --- UPDATED ARGUMENTS ---
    parser.add_argument('--target-ratio', type=float, default=0.1, 
                    help='Target ratio of Reg Grad Norm vs Task Grad Norm')
    parser.add_argument('--pulse-batches', type=int, default=10, help='Batches for PGD Pulse')
    
    args = parser.parse_args()

    # Logger Setup
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"{args.name}.log")
    sys.stdout = TeeLogger(log_file)
    
    print(f"======================================================")
    print(f" STARTING EXPERIMENT (CIFAR-100): {args.name}")
    print(f" Strategy: Dynamic Gradient Balancing | Ratio: {args.target_ratio}")
    print(f"======================================================")
    
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data - CIFAR100
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), 
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=True, download=True, transform=transform_train), 
        batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=False, transform=transform_test), 
        batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model - Swin-Tiny Optimized for CIFAR-100
    model = LipsFormerSwin(
        img_size=32, 
        patch_size=4, 
        in_chans=3, 
        num_classes=100,
        embed_dim=96,           
        depths=[2, 2, 6, 2],    
        num_heads=[3, 6, 12, 24], 
        window_size=4,          
        mlp_ratio=4.            
    ).to(device)

    # Pre-calculate Target CDF
    win_sq = 4 * 4
    ranks = torch.arange(1, win_sq + 1, dtype=torch.float32, device=device)
    target_pdf = 1.0 / (ranks ** 1.2) 
    target_pdf = target_pdf / target_pdf.sum()
    target_cdf = torch.cumsum(target_pdf, dim=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_pgd_acc = 0.0
    best_model_path = os.path.join(args.output_dir, f'{args.name}_best_pulse.pth')
    final_model_path = os.path.join(args.output_dir, f'{args.name}_final.pth')

    # --- Training Loop ---
    for epoch in range(1, args.epochs + 1):
        
        # CORRECTED CALL: Pass args.target_ratio, NOT current_lambda
        train(args, model, device, train_loader, optimizer, epoch, criterion, target_cdf, target_ratio=args.target_ratio)
        
        # Visualization (Wrapped in try/except so it doesn't crash the run)
        mid_epoch = args.epochs // 2
        if epoch == 1 or epoch == mid_epoch or epoch == args.epochs:
            try:
                save_attention_plots(model, test_loader, device, epoch, args, target_cdf)
            except Exception as e:
                print(f"Warning: Plotting failed ({e}), continuing training...")

        scheduler.step()

        # --- Pulse Check Evaluation ---
        # Check every 5 epochs, or every 2 epochs during mid-training
        if epoch % 5 == 0 or (epoch > 15 and epoch < 45 and epoch % 2 == 0) or epoch == args.epochs:
            acc, acc_pgd = test(model, device, test_loader, criterion, 
                              pgd_steps=10, desc="Pulse Check", limit_batches=args.pulse_batches)
            
            if acc_pgd > best_pgd_acc:
                best_pgd_acc = acc_pgd
                print(f"--> New Best Pulse PGD: {best_pgd_acc:.2f}% | Saving...")
                torch.save(model.state_dict(), best_model_path)

    # --- Save Final Model ---
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining Complete. Final model saved to {final_model_path}")

    # --- Final Evaluation Suite ---
    checkpoints_to_test = [
        ("FINAL MODEL", final_model_path),
        ("BEST PULSE MODEL", best_model_path)
    ]

    for label, path in checkpoints_to_test:
        if os.path.exists(path):
            print("\n" + "="*60)
            print(f" EVALUATING: {label}")
            print("="*60)
            
            model.load_state_dict(torch.load(path))
            model.eval()

            # 1. PGD-100
            print(f"\n[1/2] Running PGD-100 on {label}...")
            test(model, device, test_loader, criterion, pgd_steps=100, desc=f"PGD-100 ({label})")

            # 2. AutoAttack
            if AUTOATTACK_AVAILABLE:
                print(f"\n[2/2] Running AutoAttack on {label}...")
                run_autoattack(model, test_loader, device, log_file)
            else:
                print(f"\n[2/2] AutoAttack skipped (not installed).")
        else:
            print(f"\nWarning: Checkpoint not found at {path}")


if __name__ == '__main__':
    main()
