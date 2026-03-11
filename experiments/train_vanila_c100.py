import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast

# --- NEW: Weights & Biases Import ---
import wandb

# HuggingFace Imports for Vanilla ViT
from transformers import ViTConfig, ViTForImageClassification

# Attempt to import AutoAttack
try:
    from autoattack import AutoAttack
    AUTOATTACK_AVAILABLE = True
except ImportError:
    AUTOATTACK_AVAILABLE = False
    print("Warning: AutoAttack not installed. Final robust evaluation will be skipped.")

# ------------------------------------------------------------------------------
# 1. Augmentations (Mixup & CutMix)
# ------------------------------------------------------------------------------
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def apply_mixup_cutmix(x, y, alpha_m=0.8, alpha_c=1.0):
    if np.random.rand() < 0.5:
        lam = np.random.beta(alpha_m, alpha_m)
        rand_index = torch.randperm(x.size(0)).to(x.device)
        y_a, y_b = y, y[rand_index]
        x_mixed = lam * x + (1 - lam) * x[rand_index]
        return x_mixed, y_a, y_b, lam
    else:
        lam = np.random.beta(alpha_c, alpha_c)
        rand_index = torch.randperm(x.size(0)).to(x.device)
        y_a, y_b = y, y[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x_mixed = x.clone()
        x_mixed[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        return x_mixed, y_a, y_b, lam

# ------------------------------------------------------------------------------
# 2. Scientific Metrics (Zipfian, ERank, Spectral Norm)
# ------------------------------------------------------------------------------
def compute_zipfian_loss(attn_weights, target_cdf, order=1):
    loss = 0.0
    target_base = target_cdf.view(-1)
    
    for attn in attn_weights:
        B, H, Q, K = attn.shape
        flat_attn = attn.view(-1, K) 
        sorted_weights, _ = torch.sort(flat_attn, dim=-1, descending=True)
        current_cdf = torch.cumsum(sorted_weights, dim=-1)
        
        target = target_base.unsqueeze(0).expand_as(current_cdf)
        loss += torch.nn.functional.l1_loss(current_cdf, target)

    return loss / len(attn_weights)

def get_attention_health(attn_weights):
    all_maxes, all_entropies = [], []
    with torch.no_grad():
        for attn in attn_weights:
            max_val = attn.max(dim=-1)[0].mean().item()
            p = torch.clamp(attn, min=1e-8)
            entropy = -(p * torch.log(p)).sum(dim=-1).mean().item()
            all_maxes.append(max_val)
            all_entropies.append(entropy)
            
    return sum(all_maxes)/len(all_maxes), sum(all_entropies)/len(all_entropies)

def compute_erank(attn_weights):
    all_eranks = []
    with torch.no_grad():
        for attn in attn_weights:
            B, H, Q, K = attn.shape
            matrices = attn.view(-1, Q, K)
            s = torch.linalg.svdvals(matrices.float())
            s_norm = s / (s.sum(dim=-1, keepdim=True) + 1e-8)
            erank_entropy = -(s_norm * torch.log(s_norm + 1e-8)).sum(dim=-1)
            erank = torch.exp(erank_entropy).mean().item()
            all_eranks.append(erank)
    return sum(all_eranks) / len(all_eranks)

def compute_spectral_norm(model):
    norms = []
    for name, param in model.named_parameters():
        if ('attention' in name or 'query' in name or 'key' in name or 'value' in name or 'dense' in name) and param.ndim >= 2:
            with torch.no_grad():
                s = torch.linalg.svdvals(param.float())
                norms.append(s[0].item())
    return sum(norms) / len(norms) if norms else 0.0

# ------------------------------------------------------------------------------
# 3. Training & Evaluation Core
# ------------------------------------------------------------------------------
def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, steps=10, device='cuda'):
    images = images.to(device)
    labels = labels.to(device)
    loss_fn = nn.CrossEntropyLoss()
    delta = torch.empty_like(images).uniform_(-eps, eps)
    adv_images = torch.clamp(images + delta, 0, 1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = loss_fn(outputs.logits, labels)
        grad = torch.autograd.grad(loss, adv_images)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, -eps, eps)
        adv_images = torch.clamp(images + delta, 0, 1).detach()

    return adv_images

def train(args, model, device, train_loader, optimizer, scaler, epoch, criterion, target_cdf, target_ratio=0.1):
    print(f"\n>>> EPOCH {epoch} | Dynamic Balancing (Target: {target_ratio*100}%) <<<")
    model.train()
    
    train_loss = 0.0
    current_lambda = args.lambda_reg 
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target_a, target_b, lam = apply_mixup_cutmix(data, target)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            outputs = model(data, output_attentions=True)
            output = outputs.logits
            attn_weights = outputs.attentions 
                    
            loss_main = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            raw_reg = compute_zipfian_loss(attn_weights, target_cdf, order=args.reg_order)
        
        if batch_idx % 50 == 0:
            if target_ratio == 0.0:
                 current_lambda = 0.0
            else:
                grad_main = torch.autograd.grad(loss_main, model.parameters(), retain_graph=True, allow_unused=True)
                grad_reg = torch.autograd.grad(raw_reg, model.parameters(), retain_graph=True, allow_unused=True)
                
                norm_main = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grad_main if g is not None]), 2)
                norm_reg = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grad_reg if g is not None]), 2)
    
                ideal_lambda = torch.tensor(0.0, device=device)
                if norm_reg > 1e-8:
                    ideal_lambda = (norm_main * target_ratio) / norm_reg
                    
                current_lambda = 0.9 * current_lambda + 0.1 * ideal_lambda.item()

        total_loss = loss_main + (current_lambda * raw_reg)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += total_loss.item()
        
        if batch_idx % 100 == 0:
            avg_max_attn, avg_entropy = get_attention_health(attn_weights)
            print(f"Batch {batch_idx}: Lam {current_lambda:.4f} | Peak: {avg_max_attn:.3f} | Ent: {avg_entropy:.3f}")
            
            # --- W&B BATCH LOGGING ---
            wandb.log({
                "Train/Batch_Loss": total_loss.item(),
                "Train/Dynamic_Lambda": current_lambda,
                "Attention/Peak_Weight": avg_max_attn,
                "Attention/Entropy": avg_entropy,
            })

    # --- W&B EPOCH LOGGING ---
    wandb.log({
        "Train/Epoch_Avg_Loss": train_loss / len(train_loader),
        "epoch": epoch
    })
    print(f"Train End: Avg Loss: {train_loss/len(train_loader):.4f} | Final Lambda: {current_lambda:.4f}")

def test(model, device, test_loader, criterion, pgd_steps=0, desc="Eval"):
    model.eval()
    correct = 0
    correct_pgd = 0
    total = 0
    
    print(f"Running {desc}...", end=" ", flush=True)
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        current_batch_size = target.size(0)
        total += current_batch_size
        
        with torch.no_grad():
            outputs = model(data)
            output = outputs.logits 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        if pgd_steps > 0:
            adv_data = pgd_attack(model, data, target, eps=8/255, alpha=2/255, steps=pgd_steps, device=device)
            with torch.no_grad():
                output_adv = model(adv_data).logits 
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

class HuggingFaceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x).logits

def run_autoattack(model, test_loader, device):
    if not AUTOATTACK_AVAILABLE:
        print("AutoAttack not installed, skipping.")
        return

    print("\n>>> RUNNING AUTOATTACK (Standard: APGD-CE, APGD-T, FAB-T, Square) <<<")
    wrapped_model = HuggingFaceWrapper(model).to(device)
    wrapped_model.eval()
    
    all_imgs = []
    all_lbls = []
    for data, target in test_loader:
        all_imgs.append(data)
        all_lbls.append(target)
    
    x_test = torch.cat(all_imgs, dim=0)
    y_test = torch.cat(all_lbls, dim=0)
    
    adversary = AutoAttack(wrapped_model, norm='Linf', eps=8/255, version='standard')
    with torch.no_grad():
        _ = adversary.run_standard_evaluation(x_test, y_test, bs=100)
    print("AutoAttack evaluation complete.")

# ------------------------------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='experiment')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='logs_experiment')
    parser.add_argument('--lambda-reg', type=float, default=1.0) 
    parser.add_argument('--reg-order', type=int, default=1)         
    parser.add_argument('--target-ratio', type=float, default=0.1, help='Target ratio of Reg Grad Norm vs Task Grad Norm')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- W&B INITIALIZATION ---
    wandb.init(
        project="cifar100-vit-zipfian", 
        name=args.name,
        config=vars(args) 
    )
    
    print(f"======================================================")
    print(f" STARTING EXPERIMENT (CIFAR-100): {args.name}")
    print(f" Strategy: ViT-Tiny | RandAugment+Mixup+CutMix | Target: {args.target_ratio}")
    print(f"======================================================")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler('cuda')
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), 
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=True, download=True, transform=transform_train), 
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=False, transform=transform_test), 
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    config = ViTConfig(
        image_size=32, 
        patch_size=4, 
        num_channels=3, 
        num_labels=100,
        hidden_size=192, 
        num_hidden_layers=12, 
        num_attention_heads=3,
        attn_implementation="eager"
    )
    model = ViTForImageClassification(config).to(device)

    sequence_length = (config.image_size // config.patch_size)**2 + 1 
    ranks = torch.arange(1, sequence_length + 1, dtype=torch.float32, device=device)
    target_pdf = 1.0 / (ranks ** 1.2) 
    target_pdf = target_pdf / target_pdf.sum()
    target_cdf = torch.cumsum(target_pdf, dim=0)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    warmup_epochs = 10
    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])

    best_acc = 0.0
    best_model_path = os.path.join(args.output_dir, f'{args.name}_best.pth')

    # --- Training Loop ---
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, scaler, epoch, criterion, target_cdf, target_ratio=args.target_ratio)
        scheduler.step()

        acc, _ = test(model, device, test_loader, criterion, pgd_steps=0, desc="Test Acc")
        spec_norm = compute_spectral_norm(model)
        
        model.eval()
        sample_data, _ = next(iter(test_loader))
        with torch.no_grad():
            outputs = model(sample_data.to(device), output_attentions=True)
            epoch_erank = compute_erank(outputs.attentions)
        
        print(f"--- Epoch {epoch} Metrics ---")
        print(f"Top-1 Accuracy: {acc:.2f}%")
        print(f"Avg Spectral Norm: {spec_norm:.4f}")
        print(f"Avg Attention ERank: {epoch_erank:.2f}")
        
        # --- W&B EVAL LOGGING ---
        wandb.log({
            "Eval/Top1_Accuracy": acc,
            "Eval/Spectral_Norm": spec_norm,
            "Attention/Epoch_ERank": epoch_erank,
            "epoch": epoch
        })
        
        if acc > best_acc:
            best_acc = acc
            print(f"--> New Best Accuracy: {best_acc:.2f}% | Saving...")
            torch.save(model.state_dict(), best_model_path)
            # Log the best checkpoint to wandb directly
            wandb.save(best_model_path)

    # --- Final Evaluation Suite ---
    print("\n" + "="*60)
    print(f" FINAL EVALUATION: Best Model ({best_acc:.2f}%)")
    print("="*60)
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        print(f"\n[1/2] Running PGD-100 on Best Model...")
        _, acc_pgd100 = test(model, device, test_loader, criterion, pgd_steps=100, desc="PGD-100 Eval")
        wandb.run.summary["Final_PGD100_Acc"] = acc_pgd100 

        if AUTOATTACK_AVAILABLE:
            print(f"\n[2/2] Running AutoAttack on Best Model...")
            run_autoattack(model, test_loader, device)
            # AutoAttack prints robust acc natively, but we can signal completion to W&B
            wandb.run.summary["AutoAttack_Complete"] = True
        else:
            print(f"\n[2/2] AutoAttack skipped (not installed).")
            
    # Finish the W&B Run
    wandb.finish()

if __name__ == '__main__':
    main()
