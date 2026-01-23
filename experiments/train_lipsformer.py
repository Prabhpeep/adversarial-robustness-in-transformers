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
from models.lipsformer_swin_jacsmooth import LipsFormerSwin

def pgd_attack(model, images, labels, eps=36/255, alpha=36/255/4, steps=10, device='cuda'):
    """
    Standard PGD attack.
    """
    images = images.to(device)
    labels = labels.to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Random start
    adv_images = images.clone().detach() + torch.empty_like(images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, 0, 1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = loss_fn(outputs, labels)

        grad = torch.autograd.grad(loss, adv_images)[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, -eps, eps)
        adv_images = torch.clamp(images + delta, 0, 1).detach()

    return adv_images

def train(args, model, device, train_loader, optimizer, epoch, criterion):
    print(f"\n>>> EPOCH {epoch} | Mode: Faithful JaSMin | Lambda: {args.jasmin_lambda} <<<")
    model.train()
    train_loss = 0.0
    correct = 0
    epoch_jasmin_max = -float("inf")

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # === JaSMin Injection ===
        if args.jasmin_lambda > 0:
            reg_loss = model.jasmin_loss()
            # reg_loss is negative log-bound. Minimizing it pushes g1 -> 0.
            loss = loss + args.jasmin_lambda * reg_loss
            
            with torch.no_grad():
                if reg_loss.item() > epoch_jasmin_max:
                    epoch_jasmin_max = reg_loss.item()

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.4f}")

    train_loss /= len(train_loader.dataset)
    acc = 100. * correct / len(train_loader.dataset)
    
    print(f"Train set: Avg Loss: {train_loss:.4f}, Acc: {acc:.2f}%, JaSMin(Worst): {epoch_jasmin_max:.4f}")

def test(model, device, test_loader, criterion, pgd_steps=10, desc="Eval"):
    model.eval()
    test_loss = 0
    correct = 0
    correct_pgd = 0
    
    print(f"Running {desc} (Clean + PGD-{pgd_steps})...")
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # 1. Clean
        with torch.no_grad():
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        # 2. PGD
        if pgd_steps > 0:
            # Using eps=36/255 (~0.14) and alpha=eps/4
            eps = 36/255
            alpha = eps/4
            adv_data = pgd_attack(model, data, target, eps=eps, alpha=alpha, steps=pgd_steps, device=device)
            with torch.no_grad():
                output_adv = model(adv_data)
                pred_adv = output_adv.argmax(dim=1, keepdim=True)
                correct_pgd += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    acc_pgd = 100. * correct_pgd / len(test_loader.dataset)
    
    print(f"{desc} Results: Loss: {test_loss:.4f} | Clean Acc: {acc:.2f}% | PGD-{pgd_steps} Acc: {acc_pgd:.2f}%")
    return acc, acc_pgd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--jasmin-lambda', type=float, default=0.01) # Default lowered to 0.01
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='output_jasmin_experiment')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

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

    # Model Init
    model = LipsFormerSwin(
        img_size=32, 
        patch_size=4, 
        in_chans=3, 
        num_classes=10,
        embed_dim=96, 
        depths=[2, 2, 6, 2], 
        num_heads=[3, 6, 12, 24],
        window_size=4,
        mlp_ratio=4.
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    best_pgd_acc = 0.0
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    
    # --- Training Loop ---
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        
        # Monitor with PGD-10 (Fast)
        acc, acc_pgd = test(model, device, test_loader, criterion, pgd_steps=10, desc="Epoch Eval")
        
        scheduler.step()

        if acc_pgd > best_pgd_acc:
            best_pgd_acc = acc_pgd
            print(f"--> New Best PGD-10 Accuracy: {best_pgd_acc:.2f}% | Saving Model...")
            torch.save(model.state_dict(), best_model_path)

    # --- Final Evaluation (PGD-100) ---
    print("\n\n>>> TRAINING COMPLETE. LOADING BEST MODEL FOR PGD-100... <<<")
    model.load_state_dict(torch.load(best_model_path))
    
    test(model, device, test_loader, criterion, pgd_steps=100, desc="FINAL ROBUSTNESS CHECK")

if __name__ == '__main__':
    main()
