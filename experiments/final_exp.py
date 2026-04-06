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
# 2. Phase 1 Fixed Regularizers (Attention Geometry)
# ------------------------------------------------------------------------------
class AttentionGeometryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def extract_cls_attention(self, attn_weights):
        """
        Isolates [CLS] token's query to image patch keys and projects 
        it back onto the probability simplex.
        """
        # Slice out the CLS token's attention to the image patches
        cls_attn = attn_weights[:, :, 0, 1:]
        
        # CRITICAL FIX: Re-normalize so the patch probabilities sum to 1.0
        return cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-9)

    def compute_entropy(self, cls_attn):
        # cls_attn is now guaranteed to be a valid probability distribution
        p = torch.clamp(cls_attn, min=1e-9)
        return -(p * torch.log(p)).sum(dim=-1).mean()

    def compute_erank(self, cls_attn):
        mean_attn = cls_attn.mean(0)  
        S = torch.linalg.svdvals(mean_attn)
        p = S / (S.sum() + 1e-9)
        p = torch.clamp(p, min=1e-9)
        return torch.exp(-(p * torch.log(p)).sum()).item()

class ZipfianCDFLoss(AttentionGeometryLoss):
    def __init__(self, s):
        super().__init__()
        self.s = s
        self._cached_cdf = {} 

    def _get_ideal_cdf(self, K, device):
        key = (K, str(device))
        if key not in self._cached_cdf:
            ranks = torch.arange(1, K + 1, dtype=torch.float32, device=device)
            pmf = 1.0 / (ranks ** self.s)
            pmf /= pmf.sum()
            self._cached_cdf[key] = torch.cumsum(pmf, dim=-1)
        return self._cached_cdf[key]

    def forward(self, attn_weights):
        cls_attn = self.extract_cls_attention(attn_weights)
        B, H, K_minus_1 = cls_attn.shape
        
        ideal_cdf = self._get_ideal_cdf(K_minus_1, cls_attn.device)
        
        sorted_attn, _ = torch.sort(cls_attn, descending=True, dim=-1)
        empirical_cdf = torch.cumsum(sorted_attn, dim=-1)
        
        return F.l1_loss(empirical_cdf, ideal_cdf.expand_as(empirical_cdf))

class MarginLoss(AttentionGeometryLoss):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, attn_weights):
        cls_attn = self.extract_cls_attention(attn_weights)
        top2, _ = torch.topk(cls_attn, 2, dim=-1)
        p_max = top2[:, :, 0]
        p_2nd = top2[:, :, 1]
        
        loss = F.relu(self.beta - (p_max - p_2nd))
        return loss.mean()

class TopKSparseAttentionLoss(AttentionGeometryLoss):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, attn_weights):
        cls_attn = self.extract_cls_attention(attn_weights)
        topk_weights, _ = torch.topk(cls_attn, self.k, dim=-1)
        topk_sum = torch.sum(topk_weights, dim=-1)
        return (1.0 - topk_sum).mean()

# ------------------------------------------------------------------------------
# 3. Training & Evaluation Core (Phase 4 & 5)
# ------------------------------------------------------------------------------
def compute_val_entropy(model, attention_hook, geom_utils, device, loader, max_batches=20):
    """Phase 4 Monitoring Gate on clean validation set."""
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

def pgd_attack(model, images, labels, eps=4/255, step_size=1/255, steps=20, device='cuda'):
    """Denormalize to pixel space [0, 1] for valid clipping."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    
    images, labels = images.to(device), labels.to(device)
    
    # Denormalize to [0,1] pixel space for valid clamping
    images_px = images * std + mean
    
    delta = torch.empty_like(images_px).uniform_(-eps, eps)
    adv_px = torch.clamp(images_px + delta, 0, 1).detach()
    
    for _ in range(steps):
        adv_px.requires_grad = True
        
        # Re-normalize before forward pass
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
            data = pgd_attack(model, data, target, eps=4/255, step_size=1/255, steps=pgd_steps, device=device)
            
        with torch.no_grad():
            outputs = model(data)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += batch_size

    acc = 100. * correct / total if total > 0 else 0
    print(f"| Accuracy: {acc:.2f}% ({total} images)")
    return acc

# ------------------------------------------------------------------------------
# 4. Main Execution Setup
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-name', type=str, required=True, help="WandB Run Name")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to ImageNet-100 Clean")
    parser.add_argument('--imagenetc-dir', type=str, required=True, help="Path to ImageNet-C-100 Corrupt")
    parser.add_argument('--output-dir', type=str, default='./logs_thesis')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--reg-type', type=str, choices=['baseline', 'zipfian', 'margin', 'topk'], default='baseline')
    parser.add_argument('--reg-lambda', type=float, default=1.0)
    parser.add_argument('--zipfian-s', type=float, default=1.5)
    parser.add_argument('--margin-beta', type=float, default=0.2)
    parser.add_argument('--topk-k', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    wandb.init(project="attention-geometry-thesis", name=args.job_name, config=vars(args))
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = timm.create_model('deit_small_patch16_224', pretrained=True, attn_drop_rate=0.0).to(device)
    model.head = nn.Linear(model.head.in_features, 100).to(device)

    attention_hook = AttentionCapture(model)
    scaler = GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    if args.reg_type == 'zipfian':
        reg_module = ZipfianCDFLoss(s=args.zipfian_s)
    elif args.reg_type == 'margin':
        reg_module = MarginLoss(beta=args.margin_beta)
    elif args.reg_type == 'topk':
        reg_module = TopKSparseAttentionLoss(k=args.topk_k)
    else:
        reg_module = None

    geom_utils = AttentionGeometryLoss()

    baseline_path = os.path.join(args.output_dir, 'baseline_clean_acc.npy')
    if args.reg_type != 'baseline':
        if os.path.exists(baseline_path):
            baseline_clean_acc = float(np.load(baseline_path)[0])
            print(f"Loaded Baseline Clean Acc: {baseline_clean_acc:.2f}%")
        else:
            print("WARNING: baseline_clean_acc.npy not found. Early stopping disabled.")
            baseline_clean_acc = None
    else:
        baseline_clean_acc = None

    best_clean_acc = 0.0
    early_stop_triggered = False
    global_step = 0 

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for epoch in range(1, args.epochs + 1):
        if early_stop_triggered:
            break
            
        model.train()
        epoch_loss = 0.0
        reg_timings_ms = []

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
                    track_time = (50 <= batch_idx < 150)
                    if track_time: 
                        torch.cuda.synchronize() 
                        start_event.record()
                    
                    layer_losses = [reg_module(attn) for attn in attn_maps]
                    loss_reg = sum(layer_losses) / len(layer_losses)
                    total_loss = loss_main + (args.reg_lambda * loss_reg)
                    
                    if track_time:
                        end_event.record()
                        torch.cuda.synchronize()
                        reg_timings_ms.append(start_event.elapsed_time(end_event))

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    cls_attns = [geom_utils.extract_cls_attention(a) for a in attn_maps]
                    log_dict = {"Train/Total_Loss": total_loss.item()} 

                    if reg_module is not None:
                        log_dict["Train/Reg_Loss"] = loss_reg.item()
                        log_dict["Train/Main_Loss"] = loss_main.item()
                        log_dict["Train/Lambda_Effective"] = (args.reg_lambda * loss_reg).item()

                    if cls_attns:
                        entropies = [geom_utils.compute_entropy(c).item() for c in cls_attns]
                        mean_entropy = sum(entropies) / len(entropies)
                        log_dict["Attention/Mean_Entropy"] = mean_entropy
                        
                        for i, c in enumerate(cls_attns):
                            log_dict[f"Attention/Entropy_Layer_{i}"] = entropies[i]
                            log_dict[f"Attention/PeakWeight_Layer_{i}"] = c.max(dim=-1)[0].mean().item()
                            log_dict[f"Attention/ERank_Layer_{i}"] = geom_utils.compute_erank(c)
                            
                        print(f"E{epoch} B{batch_idx} | Total Loss: {total_loss.item():.3f} | Mean Ent: {mean_entropy:.2f}")
                        
                    wandb.log(log_dict, step=global_step)
                
            epoch_loss += total_loss.item()
            global_step += 1

        scheduler.step()
        
        if reg_timings_ms:
            avg_reg_time = sum(reg_timings_ms) / len(reg_timings_ms)
            wandb.log({"System/Forward_Pass_Cuda_Time": avg_reg_time}, step=global_step)

        # Phase 4 Validation Monitoring Gate
        val_layer_entropies = compute_val_entropy(model, attention_hook, geom_utils, device, val_loader)
        val_ent_dict = {f"Validation/Entropy_Layer_{i}": ent for i, ent in enumerate(val_layer_entropies)}
        val_ent_dict["Validation/Mean_Entropy"] = val_layer_entropies.mean()
        wandb.log(val_ent_dict, step=global_step)

        # ----------------------------------------------------------------------
        # Phase 4 Automated Gate Check (Epoch 2)
        # ----------------------------------------------------------------------
        baseline_ent_path = os.path.join(args.output_dir, 'baseline_val_entropy.npy')
        
        if args.reg_type == 'baseline' and epoch == 2:
            np.save(baseline_ent_path, np.array([val_layer_entropies.mean()]))
            print(f"Saved Baseline Val Entropy at Epoch 2: {val_layer_entropies.mean():.2f} nats")
            
        elif args.reg_type == 'zipfian' and epoch == 2:
            if os.path.exists(baseline_ent_path):
                baseline_ent = float(np.load(baseline_ent_path)[0])
                mean_val_ent = val_layer_entropies.mean()
                
                if abs(mean_val_ent - baseline_ent) < 0.1:  
                    print(f"\n[!] WARNING: Phase 4 Gate Failure Detected [!]")
                    print(f"Val entropy ({mean_val_ent:.2f}) is too close to baseline ({baseline_ent:.2f}).")
                    print(f"The regularizer is inactive. Consider increasing --reg-lambda.")
                    wandb.log({"System/Gate_Warning": 1}, step=global_step)
            else:
                print("WARNING: baseline_val_entropy.npy not found. Phase 4 Gate check skipped.")
        # ----------------------------------------------------------------------

        # Phase 5 Evaluations
        clean_acc = evaluate(model, device, val_loader, desc=f"Epoch {epoch} Clean Eval")
        wandb.log({"Performance/Clean_Validation_Acc": clean_acc}, step=global_step)

        if epoch % 2 == 0:
            pgd10_acc = evaluate(model, device, val_loader, desc=f"Epoch {epoch} PGD-10", pgd_steps=10, max_samples=1000)
            wandb.log({"Performance/PGD10_Acc": pgd10_acc}, step=global_step)

        if args.reg_type == 'baseline':
            pass 
        elif baseline_clean_acc is not None and (baseline_clean_acc - clean_acc) > 2.0:
            print(f"EARLY STOPPING: Clean accuracy dropped > 2% below baseline.")
            wandb.log({"System/Early_Stop": 1}, step=global_step)
            early_stop_triggered = True

        if clean_acc > best_clean_acc:
            best_clean_acc = clean_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'{args.job_name}_best.pth'))

    if args.reg_type == 'baseline':
        np.save(baseline_path, np.array([best_clean_acc]))
        print(f"Saved Baseline Clean Acc: {best_clean_acc:.2f}% to disk.")

    # Phase 5 Final Dual Evaluation
    print("\n--- Running Final Phase 5 Evaluations ---")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, f'{args.job_name}_best.pth'), weights_only=True))
    
    final_clean_acc = evaluate(model, device, val_loader, desc="Final Clean Eval (Best Ckpt)")
    wandb.run.summary["Final_Clean_Acc"] = final_clean_acc
    
    pgd20_acc = evaluate(model, device, val_loader, desc="Final PGD-20", pgd_steps=20)
    wandb.run.summary["Final_PGD20_Acc"] = pgd20_acc
    
    for sev in [3, 5]:
        inc_path = os.path.join(args.imagenetc_dir, f'sev{sev}')
        if os.path.exists(inc_path):
            inc_dataset = datasets.ImageFolder(inc_path, transform=transform_test)
            
            assert len(inc_dataset.classes) == 100, (
                f"ImageNet-C loader found {len(inc_dataset.classes)} classes at {inc_path}. "
                "Expected 100 — verify the severity-level directory was pre-flattened correctly."
            )
            
            inc_loader = torch.utils.data.DataLoader(
                inc_dataset, batch_size=args.batch_size, shuffle=False)
            inc_acc = evaluate(model, device, inc_loader, desc=f"Final ImageNet-C (Sev {sev})")
            wandb.run.summary[f"Final_ImageNetC_Sev{sev}_Acc"] = inc_acc
        else:
            print(f"Skipping ImageNet-C Sev {sev} (Path not found: {inc_path})")

    wandb.finish()

if __name__ == '__main__':
    main()
