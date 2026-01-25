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
import os
import torch
import numpy as np

def save_attention_plots(model, loader, device, epoch, args, target_cdf):
    """
    Saves a clean visualization of Model Attention vs Zipfian Target.
    Path format: {output_dir}/{experiment_name}_attn_epoch_{epoch}.png
    """
    model.eval()
    
    # Get a single batch
    imgs, _ = next(iter(loader)) 
    imgs = imgs.to(device)
    
    with torch.no_grad():
        _, attn_weights = model(imgs, return_attn=True)
        
    # Handle list of layers vs single tensor
    if isinstance(attn_weights, list):
        # Stack all layers: [Batch, Layers, Heads, WinSize, WinSize]
        # We just flatten everything anyway
        all_attn = torch.cat(attn_weights, dim=1) 
    else:
        all_attn = attn_weights

    # Flatten [Batch, Heads, Windows, SeqLen] -> [Total_Samples, SeqLen]
    # We want to see the distribution over the LAST dimension (Sequence/Keys)
    flat_attn = all_attn.view(-1, all_attn.shape[-1])
    
    # Sort descending to get the "Shape" profile
    sorted_attn, _ = torch.sort(flat_attn, dim=-1, descending=True)
    
    # Calculate Mean Curve across all heads/images
    mean_curve = sorted_attn.mean(dim=0).cpu().numpy()
    
    # Convert Target CDF back to PDF for plotting (Target Curve)
    target_pdf = np.diff(target_cdf.cpu().numpy(), prepend=0)

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Standard Scale (The "Shape")
    axes[0].plot(mean_curve, 'b-', linewidth=2, label='Model Actual')
    axes[0].plot(target_pdf, 'r--', linewidth=2, label='Zipfian Target', alpha=0.7)
    axes[0].set_title(f"Attention Shape (Epoch {epoch})")
    axes[0].set_xlabel("Token Rank")
    axes[0].set_ylabel("Probability Mass")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Log-Log Scale (The "Power Law" Test)
    # If the blue line is straight, it's a true power law.
    axes[1].loglog(mean_curve, 'b-', linewidth=2, label='Model Actual')
    axes[1].loglog(target_pdf, 'r--', linewidth=2, label='Target')
    axes[1].set_title("Log-Log Scale (Power Law Check)")
    axes[1].set_xlabel("Log Rank")
    axes[1].grid(True, alpha=0.3)
    
    # Neat Filename: experiment_name + epoch
    save_name = f"{args.name}_attn_ep{epoch:02d}.png"
    save_path = os.path.join(args.output_dir, save_name)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150) # Higher DPI for neatness
    plt.close()
    
    print(f"   [Visual] Saved attention plot to {save_path}")

def compute_zipfian_loss(attn_weights, target_cdf, order=1):
    """
    Computes Wasserstein distance between attention distribution and Zipfian target.
    order=1: L1 Distance (Standard)
    order=2: L2 Distance (Weierstrass-Squared / Snap-to-Grid)
    """
    loss_val = 0.0
    count = 0
    
    # Handle list of lists (from multiple layers)
    if not isinstance(attn_weights, (list, tuple)):
        attn_weights = [attn_weights]

    for layer_attn in attn_weights:
        # layer_attn shape: [Batch*NumWindows, Heads, WinSize, WinSize]
        # We flatten to observe the distribution over the keys (last dim)
        B, H, S, _ = layer_attn.shape
        flat_attn = layer_attn.view(-1, S)
        
        # Sort Descending to get the "Shape"
        sorted_attn, _ = torch.sort(flat_attn, dim=-1, descending=True)
        current_cdf = torch.cumsum(sorted_attn, dim=-1)
        
        # Ensure target on correct device
        if target_cdf.device != current_cdf.device:
            target_cdf = target_cdf.to(current_cdf.device)
            
        target = target_cdf.unsqueeze(0).expand_as(current_cdf)
        
        if order == 1:
            loss_val += torch.mean(torch.abs(current_cdf - target))
        elif order == 2:
            loss_val += torch.mean(torch.pow(current_cdf - target, 2))
            
        count += 1

    return loss_val / count if count > 0 else torch.tensor(0.0, device=target_cdf.device)

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

def train(args, model, device, train_loader, optimizer, epoch, criterion, target_cdf):
    print(f"\n>>> EPOCH {epoch} | Lambda: {args.lambda_reg} | Order: {args.reg_order} | Margin: {args.use_margin} | Noise: {args.use_noise} <<<")
    model.train()
    
    train_loss = 0.0
    reg_loss_track = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # === 1. NOISE INJECTION (Backup Strategy) ===
        # If enabled, train on slightly noisy images to force local stability
        if args.use_noise:
            # Uniform noise in [-8/255, 8/255]
            noise = (torch.rand_like(data) * 2 - 1) * (8/255)
            data = torch.clamp(data + noise, 0, 1)
        # ============================================

        optimizer.zero_grad()
        
        # === 2. FORWARD PASS ===
        # return_attn=True is crucial: we need raw attention scores for the regularizer
        output, attn_weights = model(data, return_attn=True)

        # === 3. TASK LOSS ===
        if args.use_margin:
            # Combined: CrossEntropy + Lipschitz Margin (ensures gap between class scores)
            # margin=0.3 is a standard starting point for robust training
            loss_main = criterion(output, target) + lipschitz_margin_loss(output, target, margin=0.3)
        else:
            # Standard CrossEntropy
            loss_main = criterion(output, target)

        # === 4. ZIPFIAN REGULARIZATION ===
        # We calculate the raw penalty first to log it, then scale by lambda
        raw_reg = torch.tensor(0.0, device=device)
        
        if args.lambda_reg > 0:
            # Compute Wasserstein distance (L1 or L2 based on args.reg_order)
            raw_reg = compute_zipfian_loss(attn_weights, target_cdf, order=args.reg_order)
            loss = loss_main + (args.lambda_reg * raw_reg)
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
        
        # Print status every 100 batches
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Total Loss {loss.item():.4f} | Reg Raw: {raw_reg.item():.5f} | Task Loss: {loss_main.item():.4f}")

    # End of Epoch Summary
    train_loss /= len(train_loader)
    avg_reg = reg_loss_track / len(train_loader)
    acc = 100. * correct / total
    
    print(f"Train End: Avg Loss: {train_loss:.4f} | Avg Reg: {avg_reg:.5f} | Acc: {acc:.2f}%")
  
def test(model, device, test_loader, criterion, pgd_steps=0, desc="Eval", limit_batches=None):
    """
    Evaluates model on Clean and PGD data.
    limit_batches: If set (e.g., 2), stops after 2 batches to save time (Pulse Check).
    """
    model.eval()
    test_loss = 0
    correct = 0
    correct_pgd = 0
    total = 0
    
    # Print header without newline so we can append results
    print(f"Running {desc}...", end=" ", flush=True)
    
    for batch_idx, (data, target) in enumerate(test_loader):
        # --- OPTIMIZATION: BREAK EARLY FOR PULSE CHECKS ---
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        # ------------------------------------------------
            
        data, target = data.to(device), target.to(device)
        current_batch_size = target.size(0)
        total += current_batch_size
        
        # 1. Clean Evaluation
        with torch.no_grad():
            output = model(data)
            test_loss += criterion(output, target).item() * current_batch_size
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        # 2. PGD Evaluation (Only if requested)
        if pgd_steps > 0:
            # Generate attacks (gradients are needed here, even in eval mode!)
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
    
    # --- ARGUMENTS ---
    parser.add_argument('--lambda-reg', type=float, default=0.0)    
    parser.add_argument('--reg-order', type=int, default=1)         
    parser.add_argument('--use-margin', action='store_true')        
    parser.add_argument('--use-noise', action='store_true')         
    
    args = parser.parse_args()

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

    # --- MODEL INIT (Nano Configuration) ---
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

    os.makedirs(args.output_dir, exist_ok=True)
    best_pgd_acc = 0.0
    best_model_path = os.path.join(args.output_dir, f'{args.name}_best.pth')
    
    # --- Training Loop ---
    for epoch in range(1, args.epochs + 1):
        # 1. Train (Passing target_cdf is CRITICAL)
        train(args, model, device, train_loader, optimizer, epoch, criterion, target_cdf)
        
        # ... inside the loop in main() ...
    
        # 2. Visualization Strategy: Start, Middle, End
        # For 30 epochs, this saves at 1, 15, and 30.
        mid_epoch = args.epochs // 2
        if epoch == 1 or epoch == mid_epoch or epoch == args.epochs:
            save_attention_plots(model, test_loader, device, epoch, args, target_cdf)

        scheduler.step()

        # 3. Pulse Check Evaluation (Every 5 epochs)
        # We assume you updated test() to accept limit_batches
        if epoch % 5 == 0 or epoch == args.epochs:
            # Quick check on 2 batches
            acc, acc_pgd = test(model, device, test_loader, criterion, 
                              pgd_steps=10, desc="Pulse Check", limit_batches=2)
            
            # Save if it's a good run (Optional, but good for safety)
            if acc_pgd > best_pgd_acc:
                best_pgd_acc = acc_pgd
                print(f"--> New Best Pulse PGD: {best_pgd_acc:.2f}% | Saving...")
                torch.save(model.state_dict(), best_model_path)

    # --- Final Evaluation (PGD-100) ---
    print("\n\n>>> TRAINING COMPLETE. LOADING BEST MODEL FOR PGD-100... <<<")
    
    # Load best model if it exists, otherwise use current
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    
    # Run Full Test (limit_batches=None)
    test(model, device, test_loader, criterion, pgd_steps=100, desc="FINAL ROBUSTNESS CHECK", limit_batches=None)

if __name__ == '__main__':
    main()
