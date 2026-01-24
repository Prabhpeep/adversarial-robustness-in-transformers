import torch
import torch.nn as nn 
from tqdm import tqdm 
import torchattacks
# pip install autoattack
from autoattack import AutoAttack 

def evaluate_pgd(loader, model, epsilon, device, attack_type='pgd'):
    model.eval()
    
    # 1. FIX: Track total samples explicitly to avoid "i" indexing bugs
    correct = 0
    total = 0
    
    print(f"Running {attack_type} with epsilon {epsilon}...")

    if attack_type == 'autoattack':
        # AutoAttack (Standard for papers)
        # version='standard' runs APGD-CE, APGD-DLR, FAB, and Square
        adversary = AutoAttack(model, norm='L2', eps=epsilon, version='standard', device=device)
    else:
        # Standard PGD (Quick check)
        adversary = torchattacks.PGDL2(
            model,
            eps=epsilon,
            alpha=epsilon/4,  # Heuristic step size
            steps=50,         # 50 steps is better than 10 or 20 for evaluation
            random_start=True
        )

    # 2. FIX: Disable gradients globally for model weights 
    # (We only need gradients on the INPUT X, which attacks handle automatically)
    for p in model.parameters():
        p.requires_grad = False

    for X, y in tqdm(loader):
        X, y = X.to(device), y.to(device)
        
        if attack_type == 'autoattack':
            # AutoAttack returns the adversarial images directly
            X_adv = adversary.run_standard_evaluation(X, y, bs=X.shape[0])
        else:
            # PGD Generation
            X_adv = adversary(X, y)

        # Final Inference (No grad needed here)
        with torch.no_grad():
            out = model(X_adv)
            pred = out.argmax(dim=1)
            
            # 3. FIX: Simple accuracy accumulation
            correct += pred.eq(y).sum().item()
            total += X.shape[0]

        # Break after ~1000 samples for quick checks (remove for full eval)
        if total >= 1000:
            break

    acc = 100. * correct / total
    print(f"{attack_type} Accuracy: {acc:.2f}%")
    return acc
