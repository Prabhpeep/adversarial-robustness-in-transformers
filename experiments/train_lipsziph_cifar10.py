import os
import argparse 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets, transforms
import csv
import sys
import numpy as np

# Import the model
from models.lips_ziphreg import LipsFormerSwin

import matplotlib.pyplot as plt

# --- LOGGER CLASS (DUAL OUTPUT) ---
class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure it writes immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- MODIFIED PLOTTING FUNCTION (Per-Token CDF) ---
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

        # Handle Swin List output (use last layer)
        if isinstance(attn_weights, list):
            attn = attn_weights[-1] 
        else:
            attn = attn_weights

        # Shape: [Batch, Heads, Query_N, Key_N]
        # We need to handle 4D (standard) or 3D/variable shapes carefully
        if attn.dim() == 4:
            B, H, Q, K = attn.shape
            # Flatten into [Total_Tokens, Key_Dim] to sort PER TOKEN
            # We treat every query token across Batch and Heads as an independent observer
            flat_attn = attn.view(-1, K)
        else:
            # Fallback for unexpected shapes
            flat_attn = attn.view(-1, attn.shape[-1])
        
        # Sort descending per token
        sorted_attn, _ = torch.sort(flat_attn, dim=-1, descending=True)
        
        # Compute CDF per token
        current_cdfs = torch.cumsum(sorted_attn, dim=-1)
        
        # Average the curves across all tokens (Batch * Heads * Queries)
        avg_cdf_curve = current_cdfs.mean(dim=0).cpu().numpy()
        
        # Normalize (ensure it ends at 1.0)
        avg_cdf_curve = avg_cdf_curve / (avg_cdf_curve[-1] + 1e-8)

        # Plot
        n = len(avg_cdf_curve)
        x_axis = np.linspace(0, 1, n)

        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, avg_cdf_curve, label=f'Epoch {epoch} Actual', linewidth=2.5)
        
        if target_cdf is not None:
             t_np = target_cdf.view(-1).cpu().numpy()
             # Interpolate target if dimensions differ (e.g. Swin Window sizes)
             if len(t_np) != n:
                 t_x_old = np.linspace(0, 1, len(t_np))
                 t_np = np.interp(x_axis, t_x_old, t_np)
             plt.plot(x_axis, t_np, 'r--', label='Target Zipf', linewidth=2)

        plt.title(f'Attention CDF (Epoch {epoch}) - {args.name}')
        plt.xlabel('Rank')
        plt.ylabel('Cumulative Mass')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, f"{args.name}_cdf_epoch_{epoch}.png"))
        plt.close()
        print(f"Saved attention plot to {os.path.join(plot_dir, f'{args.name}_cdf_epoch_{epoch}.png')}")
        
    model.train()
    
def compute_zipfian_loss(attn_weights, target_cdf, order=1):
    """
    Computes Zipfian regularization loss. 
    Robust to target_cdf being 1D [N] or 2D [1, N].
    """
    loss = 0.0
    
    # 1. Normalize target_cdf to be 1D [Target_N] for easier handling
    if target_cdf.dim() > 1:
        target_base = target_cdf.view(-1)
    else:
        target_base = target_cdf

    for attn in attn_weights:
        # attn shape: [Batch, Heads, N, N]
        B, H, N, _ = attn.shape
        
        # Sort weights and compute CDF
        # Flatten batch and heads: [B*H, N]
        sorted_weights, _ = torch.sort(attn.view(-1, N), dim=-1, descending=True)
        current_cdf = torch.cumsum(sorted_weights, dim=-1)
        
        # 2. Resize Target if dimensions mismatch (Swin layers have different N)
        target_len = target_base.shape[0]
        
        if target_len != N:
            # Interpolate requires [Batch, Channels, Length] -> [1, 1, Target_N]
            temp_target = target_base.view(1, 1, -1)
            resized = torch.nn.functional.interpolate(
                temp_target, 
                size=N, 
                mode='linear', 
                align_corners=False
            )
            # Flatten back to 1D [N]
            resized_target = resized.view(-1)
            # Re-normalize max to 1.0
            resized_target = resized_target / (resized_target.max() + 1e-8)
        else:
            resized_target = target_base

        # 3. Expand to match current_cdf shape [B*H, N]
        # unsqueeze(0) makes it [1, N], then expand matches batch dim
        target = resized_target.unsqueeze(0).expand_as(current_cdf)

        # 4. Compute Loss
        loss += torch.nn.functional.l1_loss(current_cdf, target)

    return loss / len(attn_weights)

def lipschitz_margin_loss(logits, targets, margin=0.3):
    """
    Ensures (Correct_Score - Runner_Up_Score) > Margin
    """
    # Select correct class scores
    correct_scores = logits.gather(1, targets.view(-1, 1)).squeeze()
    
    # Select runner-up scores
    # Mask correct class with -inf so max() picks the second best
    logits_masked = logits.clone()
    logits_masked[torch.arange(logits.size(0)), targets] = -float('inf')
    runner_up_scores = logits_masked.max(dim=1)[0]
    
    # Hinge Loss
    return torch.relu(margin - (correct_scores - runner_up_scores)).mean()

def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, steps=10, device='cuda'):
    """
    Standard PGD attack for CIFAR-10.
    Supports varying steps (e.g., 10 for quick checks, 100 for final eval).
    """
    images = images.to(device)
    labels = labels.to(device)
    loss_fn = nn.CrossEntropyLoss()

    # 1. Start with random jitter within the epsilon ball
    # This prevents the attack from getting stuck in bad local minima (random start)
    delta = torch.empty_like(images).uniform_(-eps, eps)
    adv_images = torch.clamp(images + delta, 0, 1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = loss_fn(outputs, labels)

        # 2. Calculate Gradient
        grad = torch.autograd.grad(loss, adv_images)[0]

        # 3. Update Adversarial Images
        # Ascend the gradient (maximize loss)
        adv_images = adv_images.detach() + alpha * grad.sign()

        # 4. Project back into Epsilon Ball & Valid Image Range
        delta = torch.clamp(adv_images - images, -eps, eps)
        adv_images = torch.clamp(images + delta, 0, 1).detach()

    return adv_images

def train(args, model, device, train_loader, optimizer, epoch, criterion, target_cdf, current_lambda):
    print(f"\n>>> EPOCH {epoch} | Lambda: {current_lambda:.4f} | Order: {args.reg_order} | Margin: {args.use_margin} | Noise: {args.use_noise} <<<")
    model.train()
    
    train_loss = 0.0
    reg_loss_track = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # === 1. NOISE INJECTION (Backup Strategy) ===
        if args.use_noise:
            # Uniform noise in [-8/255, 8/255]
            noise = (torch.rand_like(data) * 2 - 1) * (8/255)
            data = torch.clamp(data + noise, 0, 1)

        optimizer.zero_grad()
        
        # === 2. FORWARD PASS ===
        output, attn_weights = model(data, return_attn=True)

        # === 3. TASK LOSS ===
        if args.use_margin:
            loss_main = criterion(output, target) + lipschitz_margin_loss(output, target, margin=0.3)
        else:
            loss_main = criterion(output, target)

        # === 4. ZIPFIAN REGULARIZATION ===
        raw_reg = torch.tensor(0.0, device=device)
        
        if current_lambda > 0:
            raw_reg = compute_zipfian_loss(attn_weights, target_cdf, order=args.reg_order)
            loss = loss_main + (current_lambda * raw_reg)
        else:
            loss = loss_main

        # === 5. OPTIMIZATION ===
        loss.backward()
        optimizer.step()

        # === 6. LOGGING ===
        train_loss += loss.item()
        reg_loss_track += raw_reg.item()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Total Loss {loss.item():.4f} | Reg Raw: {raw_reg.item():.5f} | Task Loss: {loss_main.item():.4f}")

    train_loss /= len(train_loader)
    avg_reg = reg_loss_track / len(train_loader)
    acc = 100. * correct / total
    
    print(f"Train End: Avg Loss: {train_loss:.4f} | Avg Reg: {avg_reg:.5f} | Acc: {acc:.2f}%")
  
def test(model, device, test_loader, criterion, pgd_steps=0, desc="Eval", limit_batches=None):
    """
    Evaluates model on Clean and PGD data.
    """
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
            adv_data = pgd_attack(model, data, target, 
                                  eps=8/255, alpha=2/255, steps=pgd_steps, device=device)
            
            with torch.no_grad():
                output_adv = model(adv_data)
                pred_adv = output_adv.argmax(dim=1, keepdim=True)
                correct_pgd += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    test_loss /= total
    acc = 100. * correct / total
    
    if pgd_steps > 0:
        acc_pgd = 100. * correct_pgd / total
        print(f"| Clean: {acc:.2f}% | PGD-{pgd_steps}: {acc_pgd:.2f}% ({total} images)")
        return acc, acc_pgd
    else:
        print(f"| Clean: {acc:.2f}% ({total} images)")
        return acc, 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='experiment')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='logs_experiment')
    
    # --- EXISTING ARGUMENTS ---
    parser.add_argument('--lambda-reg', type=float, default=0.0)    
    parser.add_argument('--reg-order', type=int, default=1)         
    parser.add_argument('--use-margin', action='store_true')        
    parser.add_argument('--use-noise', action='store_true') 
    parser.add_argument('--warmup-start', type=int, default=5, help='Epoch to start ramping up lambda')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Epoch to reach full lambda')

    # --- NEW DECAY ARGUMENTS ---
    parser.add_argument('--lambda-min', type=float, default=0.0, help='Minimum Lambda after decay')
    parser.add_argument('--decay-start', type=int, default=15, help='Epoch to start decaying lambda')
    
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    # --- LOGGER SETUP ---
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"{args.name}.log")
    sys.stdout = TeeLogger(log_file)
    
    print(f"======================================================")
    print(f" STARTING EXPERIMENT: {args.name}")
    print(f" Logs saved to: {log_file}")
    print(f"======================================================")
    
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} | Experiment: {args.name}")
    
    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --- MODEL INIT (Kept exactly as requested) ---
    model = LipsFormerSwin(
        img_size=32, 
        patch_size=4, 
        in_chans=3, 
        num_classes=10,
        embed_dim=96,           
        depths=[2, 2, 2, 2],    
        num_heads=[3, 3, 3, 3], 
        window_size=4,          
        mlp_ratio=2.            
    ).to(device)

    # --- PRE-CALCULATE TARGET CDF ---
    win_sq = 4 * 4
    ranks = torch.arange(1, win_sq + 1, dtype=torch.float32, device=device)
    target_pdf = 1.0 / (ranks ** 1.2) 
    target_pdf = target_pdf / target_pdf.sum()
    target_cdf = torch.cumsum(target_pdf, dim=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_pgd_acc = 0.0
    best_model_path = os.path.join(args.output_dir, f'{args.name}_best.pth')

    # --- Training Loop ---
    for epoch in range(1, args.epochs + 1):
        
        # === DYNAMIC LAMBDA SCHEDULER ===
        # 1. Warmup Phase (Standard)
        if epoch < args.warmup_start:
            current_lambda = 0.0
        elif epoch < args.warmup_epochs:
            numerator = epoch - args.warmup_start
            denominator = args.warmup_epochs - args.warmup_start
            progress = numerator / max(denominator, 1) 
            current_lambda = args.lambda_reg * progress
        else:
            current_lambda = args.lambda_reg
            
        # 2. Linear Decay Logic (Overrides Warmup if applicable)
        if epoch >= args.decay_start:
            total_decay_epochs = args.epochs - args.decay_start
            if total_decay_epochs > 0:
                decay_progress = (epoch - args.decay_start) / total_decay_epochs
                # Decay from lambda_reg down to lambda_min
                current_lambda = args.lambda_reg - decay_progress * (args.lambda_reg - args.lambda_min)
                current_lambda = max(current_lambda, args.lambda_min)

        print(f"\n>>> EPOCH {epoch} | Lambda: {current_lambda:.4f} (Target: {args.lambda_reg}) | Decay Start: {args.decay_start} <<<")

        # 1. Train
        train(args, model, device, train_loader, optimizer, epoch, criterion, target_cdf, current_lambda)
        
        # 2. Visualization
        mid_epoch = args.epochs // 2
        if epoch == 1 or epoch == mid_epoch or epoch == args.epochs:
            save_attention_plots(model, test_loader, device, epoch, args, target_cdf)

        scheduler.step()

        # 3. Pulse Check Evaluation
        if epoch % 5 == 0 or epoch == args.epochs:
            acc, acc_pgd = test(model, device, test_loader, criterion, 
                              pgd_steps=10, desc="Pulse Check", limit_batches=2)
            
            if acc_pgd > best_pgd_acc:
                best_pgd_acc = acc_pgd
                print(f"--> New Best Pulse PGD: {best_pgd_acc:.2f}% | Saving...")
                torch.save(model.state_dict(), best_model_path)

    # --- Final Evaluation (PGD-100) ---
    print("\n\n>>> TRAINING COMPLETE. LOADING BEST MODEL FOR PGD-100... <<<")
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    
    test(model, device, test_loader, criterion, pgd_steps=100, desc="FINAL ROBUSTNESS CHECK", limit_batches=None)

if __name__ == '__main__':
    main()
