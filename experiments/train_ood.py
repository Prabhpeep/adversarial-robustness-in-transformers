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

# HuggingFace Imports for Vanilla ViT
from transformers import ViTConfig, ViTForImageClassification
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms

# Attempt to import AutoAttack (ensure 'pip install autoattack' is run)
try:
    from autoattack import AutoAttack
    AUTOATTACK_AVAILABLE = True
except ImportError:
    AUTOATTACK_AVAILABLE = False
    print("Warning: AutoAttack not installed. Final robust evaluation will be skipped.")

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
# 1. Plotting Utility
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
            # HuggingFace forward pass
            outputs = model(data, output_attentions=True)
            attn_weights = outputs.attentions
        except TypeError:
            return 

        if attn_weights is None: return

        # Get the attention from the final layer
        if isinstance(attn_weights, tuple) or isinstance(attn_weights, list): 
            attn = attn_weights[-1] 
        else: 
            attn = attn_weights

        # Flatten [B, H, Q, K] -> [Total_Queries, K]
        B, H, Q, K = attn.shape
        flat_attn = attn.view(-1, K)
        
        # Sort & CDF per query over keys
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
    target_base = target_cdf.view(-1)
    
    for attn in attn_weights:
        # attn shape: (B, H, Q, K). For ViT, Q == K == N
        B, H, Q, K = attn.shape
        
        # We flatten batch, heads, and queries to sort over the key distribution
        flat_attn = attn.view(-1, K) 
        sorted_weights, _ = torch.sort(flat_attn, dim=-1, descending=True)
        current_cdf = torch.cumsum(sorted_weights, dim=-1)
        
        target = target_base.unsqueeze(0).expand_as(current_cdf)
        loss += torch.nn.functional.l1_loss(current_cdf, target)

    return loss / len(attn_weights)

def get_attention_health(attn_weights):
    """Calculates max weight (peakiness) and entropy of the attention distributions."""
    all_maxes = []
    all_entropies = []
    
    with torch.no_grad():
        for attn in attn_weights:
            # attn shape: [B, H, Q, K]
            max_val = attn.max(dim=-1)[0].mean().item()
            
            # Entropy: -sum(p * log(p))
            p = torch.clamp(attn, min=1e-8)
            entropy = -(p * torch.log(p)).sum(dim=-1).mean().item()
            
            all_maxes.append(max_val)
            all_entropies.append(entropy)
            
    return sum(all_maxes)/len(all_maxes), sum(all_entropies)/len(all_entropies)

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
        loss = loss_fn(outputs.logits, labels) # Fixed: extract logits
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
    current_lambda = args.lambda_reg 
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # 1. Forward Pass (HuggingFace)
        outputs = model(data, output_attentions=True)
        output = outputs.logits
        attn_weights = outputs.attentions 
                
        # 2. Compute Task Loss (Main)
        loss_main = criterion(output, target)
        if args.use_margin:
            loss_main += lipschitz_margin_loss(output, target, margin=0.3)

        # 3. Compute Regularization Loss (Zipf)
        raw_reg = compute_zipfian_loss(attn_weights, target_cdf, order=args.reg_order)
        
        # Track Attention Health
        avg_max_attn, avg_entropy = get_attention_health(attn_weights)

        # 4. Dynamic Lambda Calibration
        if batch_idx % 50 == 0:
            if args.target_ratio == 0.0:
                 current_lambda = 0.0
                 norm_main = 0.0 
                 norm_reg = 0.0
            else:
                grad_main = torch.autograd.grad(loss_main, model.parameters(), retain_graph=True, allow_unused=True)
                norm_main = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grad_main if g is not None]), 2)
    
                grad_reg = torch.autograd.grad(raw_reg, model.parameters(), retain_graph=True, allow_unused=True)
                norm_reg = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grad_reg if g is not None]), 2)
    
                if norm_reg > 1e-8:
                    ideal_lambda = (norm_main * target_ratio) / norm_reg
                current_lambda = 0.9 * current_lambda + 0.1 * ideal_lambda.item()

        # 5. Final Combined Loss
        total_loss = loss_main + (current_lambda * raw_reg)
        total_loss.backward()
        optimizer.step()

        # Tracking
        train_loss += total_loss.item()
        reg_loss_track += raw_reg.item()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Lam {current_lambda:.4f} | Peak Attn: {avg_max_attn:.3f} | Ent: {avg_entropy:.3f}")

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
            outputs = model(data)
            output = outputs.logits # Fixed: extract logits
            test_loss += criterion(output, target).item() * current_batch_size
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        if pgd_steps > 0:
            adv_data = pgd_attack(model, data, target, eps=8/255, alpha=2/255, steps=pgd_steps, device=device)
            with torch.no_grad():
                output_adv = model(adv_data).logits # Fixed: extract logits
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

# --- WRAPPER FOR AUTOATTACK ---
class HuggingFaceWrapper(nn.Module):
    """AutoAttack expects raw tensors out, not HF ModelOutput objects."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x).logits

def run_autoattack(model, test_loader, device, log_file):
    if not AUTOATTACK_AVAILABLE:
        print("AutoAttack not installed, skipping.")
        return

    print("\n>>> RUNNING AUTOATTACK (Standard: APGD-CE, APGD-T, FAB-T, Square) <<<")
    model.eval()
    wrapped_model = HuggingFaceWrapper(model).to(device)
    wrapped_model.eval()
    
    # Collect all test data
    all_imgs = []
    all_lbls = []
    for data, target in test_loader:
        all_imgs.append(data)
        all_lbls.append(target)
    
    x_test = torch.cat(all_imgs, dim=0)
    y_test = torch.cat(all_lbls, dim=0)
    
    # Run AA (batch size 100 to avoid OOM)
    adversary = AutoAttack(wrapped_model, norm='Linf', eps=8/255, version='standard')
    
    with torch.no_grad():
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=100)
    
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
    parser.add_argument('--lambda-reg', type=float, default=1.0) 
    parser.add_argument('--reg-order', type=int, default=1)         
    parser.add_argument('--use-margin', action='store_true')        
    parser.add_argument('--use-noise', action='store_true') 

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
    
    # 1. Load FMoW Dataset
    dataset = get_dataset(dataset="fmow", download=True)
    
    # 2. Standard ViT Transforms (Resize to 224x224)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 3. Get the OOD Splits
    train_data = dataset.get_subset("train", transform=transform)
    val_ood_data = dataset.get_subset("val", transform=transform) # OOD Validation
    test_ood_data = dataset.get_subset("test", transform=transform) # OOD Test
    
    train_loader = get_train_loader("standard", train_data, batch_size=args.batch_size)
    test_loader = get_eval_loader("standard", test_ood_data, batch_size=args.batch_size)

    # Load the highly discriminative pre-trained ViT
    # FMoW has 62 classes
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k', 
        num_labels=62, 
        ignore_mismatched_sizes=True, 
        attn_implementation="eager", # Crucial for SAAR
        output_attentions=True
    ).to(device)
    
    # Update Sequence Length for the new 224x224 resolution
    # 224/16 = 14 patches per side -> 14*14 = 196 + 1 CLS token = 197
    sequence_length = 197 
    ranks = torch.arange(1, sequence_length + 1, dtype=torch.float32, device=device)
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
        
        train(args, model, device, train_loader, optimizer, epoch, criterion, target_cdf, target_ratio=args.target_ratio)
        
        # Visualization
        mid_epoch = args.epochs // 2
        if epoch == 1 or epoch == mid_epoch or epoch == args.epochs:
            try:
                save_attention_plots(model, test_loader, device, epoch, args, target_cdf)
            except Exception as e:
                print(f"Warning: Plotting failed ({e}), continuing training...")

        scheduler.step()

        # --- Pulse Check Evaluation ---
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

            print(f"\n[1/2] Running PGD-100 on {label}...")
            test(model, device, test_loader, criterion, pgd_steps=100, desc=f"PGD-100 ({label})")

            if AUTOATTACK_AVAILABLE:
                print(f"\n[2/2] Running AutoAttack on {label}...")
                run_autoattack(model, test_loader, device, log_file)
            else:
                print(f"\n[2/2] AutoAttack skipped (not installed).")
        else:
            print(f"\nWarning: Checkpoint not found at {path}")

if __name__ == '__main__':
    main()
