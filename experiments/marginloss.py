import argparse
import os
import random
import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
import wandb
import timm

# ------------------------------------------------------------------------------
# 1. Reproducibility & TIMM Attention Hooks
# ------------------------------------------------------------------------------
def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class AttentionCapture:
    def __init__(self, model):
        self.attention_maps = []
        self.hooks = []
        for block in model.blocks:
            hook = block.attn.attn_drop.register_forward_hook(self._hook_fn)
            self.hooks.append(hook)

    def _hook_fn(self, module, input, output):
        self.attention_maps.append(output)

    def clear(self):
        self.attention_maps.clear()

    def get_maps(self):
        return [a.float() for a in self.attention_maps]

    def remove(self):
        for h in self.hooks:
            h.remove()

# ------------------------------------------------------------------------------
# 2. Attention Geometry Regularizers
# ------------------------------------------------------------------------------
class AttentionGeometryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def extract_cls_attention(self, attn_weights):
        cls_attn = attn_weights[:, :, 0, 1:]
        return cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-9)

    def compute_entropy(self, cls_attn):
        p = torch.clamp(cls_attn, min=1e-9)
        return -(p * torch.log(p)).sum(dim=-1).mean()

class MarginLoss(AttentionGeometryLoss):
    def __init__(self, beta=0.2):
        super().__init__()
        self.beta = beta

    def forward(self, attn_weights):
        cls_attn = self.extract_cls_attention(attn_weights)
        top2, _ = torch.topk(cls_attn, 2, dim=-1)
        p_max = top2[:, :, 0]
        p_2nd = top2[:, :, 1]
        loss_per_head = F.relu(self.beta - (p_max - p_2nd))
        return loss_per_head.mean()

class JaSMinLoss(AttentionGeometryLoss):
    def __init__(self, use_log=False):
        super().__init__()
        self.use_log = use_log

    def forward(self, attn_weights):
        cls_attn = self.extract_cls_attention(attn_weights)
        top2, _ = torch.topk(cls_attn, 2, dim=-1)
        p_1 = top2[:, :, 0] 
        p_2 = top2[:, :, 1] 
        g1 = p_1 * (1.0 - p_1 + p_2)
        
        if self.use_log:
            loss_per_head = torch.log(g1 + 1e-8)
        else:
            loss_per_head = g1
        return loss_per_head.mean()

# ------------------------------------------------------------------------------
# 3. Evaluation Functions
# ------------------------------------------------------------------------------
def compute_val_entropy(model, attention_hook, geom_utils, device, loader, max_batches=20):
    model.eval()
    all_layer_entropies = []
    try:
        with torch.no_grad():
            for i, (data, _) in enumerate(loader):
                if i >= max_batches: break
                data = data.to(device)
                attention_hook.clear()
                model(data)
                attn_maps = attention_hook.get_maps()
                cls_attns = [geom_utils.extract_cls_attention(a) for a in attn_maps]
                all_layer_entropies.append([geom_utils.compute_entropy(c).item() for c in cls_attns])
    finally:
        attention_hook.clear()
        model.train()
    return np.mean(all_layer_entropies, axis=0) 

def pgd_attack(model, images, labels, eps=2/255, step_size=0.5/255, steps=10, device='cuda'):
    """Note: Default eps changed to 2/255 for the micro-crucible"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    
    images, labels = images.to(device), labels.to(device)
    images_px = images * std + mean
    
    delta = torch.empty_like(images_px).uniform_(-eps, eps)
    adv_px = torch.clamp(images_px + delta, 0, 1).detach()
    
    for _ in range(steps):
        adv_px.requires_grad = True
        adv_norm = (adv_px - mean) / std
        outputs = model(adv_norm)
        
        loss = nn.CrossEntropyLoss()(outputs, labels)
        grad = torch.autograd.grad(loss, adv_px)[0]
        
        adv_px = adv_px.detach() + step_size * grad.sign()
        delta = torch.clamp(adv_px - images_px, -eps, eps)
        adv_px = torch.clamp(images_px + delta, 0, 1).detach()
        
    return (adv_px - mean) / std 

def evaluate(model, device, loader, desc="Eval", pgd_steps=0, max_samples=None):
    model.eval()
    correct, total = 0, 0
    print(f"Running {desc}...", end=" ", flush=True)
    
    if max_samples:
        max_batches = (max_samples + loader.batch_size - 1) // loader.batch_size
        eval_loader = itertools.islice(loader, max_batches)
    else:
        eval_loader = loader

    for data, target in eval_loader:
        data, target = data.to(device), target.to(device)
        batch_size = target.size(0)
        
        if pgd_steps > 0:
            data = pgd_attack(model, data, target, eps=2/255, step_size=0.5/255, steps=pgd_steps, device=device)
            
        with torch.no_grad():
            outputs = model(data)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += batch_size

    acc = 100. * correct / total if total > 0 else 0
    print(f"| Accuracy: {acc:.2f}% ({total} images)")
    return acc

# ------------------------------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-name', type=str, required=True, help="WandB Run Name")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to ImageNet-100/CIFAR-100 Clean")
    parser.add_argument('--output-dir', type=str, default='./logs_thesis')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--reg-type', type=str, choices=['baseline', 'jasmin', 'margin'], default='baseline')
    parser.add_argument('--reg-lambda', type=float, default=0.1)
    parser.add_argument('--margin-beta', type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    wandb.init(project="attention-geometry-thesis", name=args.job_name, config=vars(args))
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=transform_train),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=transform_test),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model Setup
    model = timm.create_model('deit_small_patch16_224', pretrained=True, attn_drop_rate=0.0).to(device)
    model.head = nn.Linear(model.head.in_features, 100).to(device)
    attention_hook = AttentionCapture(model)
    geom_utils = AttentionGeometryLoss()

    scaler = GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Regularizer Instantiation
    if args.reg_type == 'margin':
        reg_module = MarginLoss(beta=args.margin_beta)
    elif args.reg_type == 'jasmin':
        reg_module = JaSMinLoss(use_log=False)
    else:
        reg_module = None

    global_step = 0 
    best_clean_acc = 0.0

    print(f"Starting {args.job_name} | Reg: {args.reg_type} | Lambda: {args.reg_lambda}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            attention_hook.clear() 
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(data)
                loss_main = criterion(outputs, target)
                total_loss = loss_main 
                loss_reg = torch.tensor(0.0, device=device)
                
                attn_maps = attention_hook.get_maps() 
                
                if reg_module is not None and len(attn_maps) > 0:
                    layer_losses = [reg_module(attn) for attn in attn_maps]
                    loss_reg = sum(layer_losses) / len(layer_losses)
                    total_loss = loss_main + (args.reg_lambda * loss_reg)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    cls_attns = [geom_utils.extract_cls_attention(a) for a in attn_maps]
                    log_dict = {"Train/Total_Loss": total_loss.item(), "Train/Main_Loss": loss_main.item()} 

                    if reg_module is not None:
                        log_dict["Train/Reg_Loss"] = loss_reg.item()

                    if cls_attns:
                        entropies = [geom_utils.compute_entropy(c).item() for c in cls_attns]
                        mean_entropy = sum(entropies) / len(entropies)
                        log_dict["Attention/Mean_Entropy"] = mean_entropy
                        print(f"E{epoch} B{batch_idx} | Total Loss: {total_loss.item():.3f} | Mean Ent: {mean_entropy:.2f}")
                        
                    wandb.log(log_dict, step=global_step)
                
            global_step += 1

        scheduler.step()

        # End of Epoch Evaluations
        val_layer_entropies = compute_val_entropy(model, attention_hook, geom_utils, device, val_loader)
        wandb.log({"Validation/Mean_Entropy": val_layer_entropies.mean()}, step=global_step)

        clean_acc = evaluate(model, device, val_loader, desc=f"Epoch {epoch} Clean Eval")
        wandb.log({"Performance/Clean_Validation_Acc": clean_acc}, step=global_step)

        # Run a quick PGD-10 test on a subset to track robust trajectory
        pgd10_acc = evaluate(model, device, val_loader, desc=f"Epoch {epoch} PGD-10", pgd_steps=10, max_samples=1000)
        wandb.log({"Performance/PGD10_Acc": pgd10_acc}, step=global_step)

        if clean_acc > best_clean_acc:
            best_clean_acc = clean_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'{args.job_name}_best.pth'))

    # Final robust check on the full validation set
    print("\n--- Running Final Phase 5 Evaluations ---")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, f'{args.job_name}_best.pth'), weights_only=True))
    final_pgd10_acc = evaluate(model, device, val_loader, desc="Final PGD-10 (Full Set)", pgd_steps=10)
    wandb.run.summary["Final_PGD10_Acc"] = final_pgd10_acc

    wandb.finish()

if __name__ == '__main__':
    main()
