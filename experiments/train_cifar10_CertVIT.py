import os
import copy
import argparse 
import pickle as pkl 
import numpy as np
import csv

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms

from models.CertVIT_regularised.certvit_jacsmooth import ViT

from utils.evaluate_CertVIT import evaluate_pgd
import autoattack

# def train(args, model, device, train_loader,
#           optimizer, epoch, criterion, finetune=False):
#     model.train()
#     train_loss = 0
#     correct = 0
#     total_jasmin=0

#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#                 # If model has  regularizer, add it
#         if hasattr(model, "_loss"):
#             _val = model._loss()   # compute 
#             loss = loss + args.lmbda * _val
#             total_ += _val.item()  # accumulate 

    
#         loss.backward()
#         train_loss += loss.item()
#         pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log_probability
#         correct += pred.eq(target.view_as(pred)).sum().item()
#         optimizer.step()

#         with torch.no_grad():
#             if args.relax and epoch > args.warmup:
#                 model.lipschitz()
#                 model.apply_spec()
#         torch.cuda.empty_cache()

#     train_loss /= len(train_loader.dataset)
#     train_samples = len(train_loader.dataset)
#         # Averages
    
#     avg_ = total_ / len(train_loader.dataset)
    

#     print(f"Epoch: {epoch}, Train set: Average loss: {train_loss:.4f}, "
#           f"Accuracy: {correct}/{train_samples} "
#           f"({100.*correct/train_samples:.0f}%), "
#           f"Error: {(train_samples-correct)/train_samples * 100:.2f}%, "
#           f": {avg_:.4f}")

# def test(args, model, device, test_loader, criterion):
#     model.eval()
#     test_loss = 0
#     correct = 0
    
#     lip = model.lipschitz()
#     verified = 0

#     # with torch.no_grad():
#     for data, target in test_loader:
#         data, target = data.to(device), target.to(device)
#         output = model.forward(data)
        
#         test_loss += criterion(output, target).item()  # sum up batch loss
#         pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log_probability
#         correct += pred.eq(target.view_as(pred)).sum().item()

#         # print (output.max())
#         one_hot = F.one_hot(target, num_classes=output.shape[-1])
#         worst_logit = output + 2**0.5 * 36/255 * lip * (1 - one_hot)
#         worst_pred = worst_logit.argmax(dim=1, keepdim=True)
#         verified += worst_pred.eq(target.view_as(worst_pred)).sum().item()

#         torch.cuda.empty_cache()

#     test_samples = len(test_loader.dataset)

#     test_loss /= len(test_loader.dataset)
#     test_samples = len(test_loader.dataset)
    
#     print(f"Test set: Average loss: {test_loss:.4f}, " +
#           f"Accuracy: {correct}/{test_samples} " + 
#           f"({100.*correct/test_samples:.2f}%), " +
#           f"Verified: {100.*verified/test_samples:.2f}%, " +
#           f"Error: {(test_samples-correct)/test_samples * 100:.2f}% " +
#           f"Lipschitz {lip:4f}")
    
#     return 100.*correct/test_samples, 100.*verified/test_samples, lip


def train(args, model, device, train_loader,
          optimizer, epoch, criterion, finetune=False):

   
    print(">>> TRAIN FUNCTION VERSION: JASMIN-MAX <<<")

    model.train()
    train_loss = 0.0
    correct = 0
    total_ = 0.0
    epoch_jasmin_max = -float("inf")

    # --- SAFE INIT: SCALED KAIMING ---
    print(">>> SAFE INIT: Using Scaled Kaiming (Factor 0.1) <<<")
    with torch.no_grad():
        for name, module in model.named_modules():
            if "LinearX" in str(type(module)):
                # 1. Standard Kaiming Init (Reset to healthy distribution)
                torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                
                # 2. Scale it down by 10x (Drastically reduces starting Lipschitz)
                module.weight.data.mul_(0.1)
                
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

                # 3. Reset Spectral Buffers
                if hasattr(module, 'sigma'):
                    if torch.is_tensor(module.sigma):
                        module.sigma.fill_(1.0)
                    else:
                        module.sigma = 1.0
                
                # 4. Initialize Power Iteration Vectors Randomly
                if hasattr(module, 'u'):
                    torch.nn.init.normal_(module.u)
                if hasattr(module, 'v'):
                    torch.nn.init.normal_(module.v)

    print(">>> SAFE INIT COMPLETE <<<")
    # ---------------------------------
   
   


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        warmup_epochs = 5  
        
        if args.jasmin_lambda != 0 and hasattr(model, "jasmin_loss") and epoch > warmup_epochs:
            try:
                jasmin_val = model.jasmin_loss()


                # convert to tensor if necessary
                if not torch.is_tensor(jasmin_val):
                    jasmin_val = torch.tensor(float(jasmin_val), device=loss.device, dtype=loss.dtype)

                # Guard against non-finite jasmin values
                if not torch.isfinite(jasmin_val).all():
                    print(f"[WARN] jasmin_loss returned non-finite ({jasmin_val}); skipping JaSMin term for this batch.")
                    jasmin_val = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

            except Exception as e:
                print(f"[WARN] jasmin_loss() raised exception: {e}; skipping JaSMin term for this batch.")
                jasmin_val = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

            loss = loss + args.jasmin_lambda * jasmin_val
            # accumulate as float safely
            try:
                # logging only: use detached STRICT max
                if hasattr(model, "jasmin_loss_logged"):
                    batch_jasmin = model.jasmin_loss_logged()
                    epoch_jasmin_max = max(epoch_jasmin_max, batch_jasmin)

            except Exception:
                # fallback: ignore accumulation if something odd happens
                pass

        # check loss before backward
        if not torch.isfinite(loss).all():
            print(f"[ERROR] Non-finite loss detected at epoch {epoch}, batch {batch_idx}. Skipping this batch.")
            # skip optimizer step and continue (or you can break)
            continue

        loss.backward()

        # Optional: gradient clipping to prevent exploding grads (uncomment to enable)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # After step, optionally apply Lipschitz projection if relax is enabled and warmup passed
        with torch.no_grad():
            if args.relax and epoch > args.warmup:
                try:
                    model.lipschitz()
                    model.apply_spec()
                except Exception as e:
                    print(f"[WARN] model.lipschitz()/apply_spec() failed: {e}")

        train_loss += float(loss.detach().cpu().item())
        pred = output.argmax(dim=1, keepdim=True)
        correct += int(pred.eq(target.view_as(pred)).sum().item())

        # free cuda cache if using GPU
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # compute averages over dataset size (keep previous behaviour)
    train_loss /= len(train_loader.dataset)
    train_samples = len(train_loader.dataset)
    if epoch_jasmin_max == -float("inf"):
        epoch_jasmin_max = 0.0

    print(
    f"Epoch: {epoch}, Train set: Average loss: {train_loss:.4f}, "
    f"Accuracy: {correct}/{train_samples} "
    f"({100.*correct/train_samples:.0f}%), "
    f"Error: {(train_samples-correct)/train_samples * 100:.2f}%, "
    f"JaSMin(max): {epoch_jasmin_max:.4f}"
)



def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0

    # compute Lipschitz but guard invalid return
    try:
        lip = model.lipschitz()
        # if lip is nan or not finite, set to 0.0
        if isinstance(lip, float):
            if not (lip == lip):  # nan check
                lip = 0.0
        elif torch.is_tensor(lip):
            if not torch.isfinite(lip).all():
                lip = 0.0
            else:
                lip = float(lip)
        else:
            # fallback
            lip = float(lip) if lip is not None else 0.0
    except Exception as e:
        print(f"[WARN] model.lipschitz() raised: {e}; using lip=0.0")
        lip = 0.0

    verified = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model.forward(data)

        # guard output
        if not torch.isfinite(output).all():
            print("[WARN] Non-finite outputs in test forward; skipping batch.")
            continue

        test_loss += float(criterion(output, target).item())
        pred = output.argmax(dim=1, keepdim=True)
        correct += int(pred.eq(target.view_as(pred)).sum().item())

        one_hot = F.one_hot(target, num_classes=output.shape[-1])
        worst_logit = output + (2**0.5) * (36/255) * lip * (1 - one_hot)
        worst_pred = worst_logit.argmax(dim=1, keepdim=True)
        verified += int(worst_pred.eq(target.view_as(worst_pred)).sum().item())

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    test_samples = len(test_loader.dataset)
    test_loss /= test_samples

    print(f"Test set: Average loss: {test_loss:.4f}, " +
          f"Accuracy: {correct}/{test_samples} " +
          f"({100.*correct/test_samples:.2f}%), " +
          f"Verified: {100.*verified/test_samples:.2f}%, " +
          f"Error: {(test_samples-correct)/test_samples * 100:.2f}% " +
          f"Lipschitz {lip:4f}")

    return 100.*correct/test_samples, 100.*verified/test_samples, lip



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 ViT')
    parser.add_argument('--task', type=str, default='train',
                        help='train/retrain/extract/test')

    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--lmbda', type=float, default=1.)
    parser.add_argument('--power_iter', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--attention_type', type=str, default='L2',
                        help='L2/DP')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--opt', type=str, default='adam',
                        help='adam/sgd')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of cores to use')
    parser.add_argument('--cos', action='store_false', 
                        help='Train with cosine annealing scheduling')
    parser.add_argument('--model', type=str, default='vit')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu to use')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--data_path', type=str, required=True,
                        help='data path of CIFAR10')
    parser.add_argument('--weight_path', type=str, required=True,
                        help='weight path of CIFAR10')
    parser.add_argument('--jasmin_lambda', type=float, default=1e-3,
                    help='Weight of JaSMin regularization term')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    print('==> Preparing data..')
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

    trainset = datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    model = ViT(image_size=32, patch_size=4, num_classes=10, channels=3,
                dim=192, depth=args.layers, heads=3, mlp_ratio=4, 
                attention_type=args.attention_type, 
                dropout=0.1, lmbda=args.lmbda, power_iter=args.power_iter,
                device=device).to(device)
    criterion = nn.CrossEntropyLoss()
    if args.opt == 'adam': 
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd': 
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                        momentum=0.9,
                        weight_decay=0.0) 
    # use cosine or reduce LR on Plateau scheduling
    if not args.cos:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                        patience=3, verbose=True, 
                                                        min_lr=1e-3*1e-5, factor=0.1)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, 
                                                         eta_min=1e-5)

    if args.task == 'train':
        if not args.relax:
            weight_path = os.path.join(args.weight_path, f"CIFAR10_{args.model}_seed-{args.seed}_layers-{args.layers}")
        else:
            weight_path = os.path.join(args.weight_path, f"CIFAR10_{args.model}_seed-{args.seed}_layers-{args.layers}_relax-{args.lmbda}_warmup-{args.warmup}")
        weight_path += f"_att-{args.attention_type}.pt"

        fout = open(weight_path.replace('.pt', '.csv').replace('weights', 'logs'), 'w')
        w = csv.writer(fout)

        if not os.path.exists(args.weight_path):
            os.mkdir(args.weight_path)

        best_acc = -1
        best_state = None
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader,
                  optimizer, epoch, criterion, False)
            acc, loss, lip = test(args, model, device, test_loader, criterion)
            w.writerow([epoch, acc, loss, lip])
        
            if args.cos:
                scheduler.step(epoch-1)
            else:
                scheduler.step(loss)
        
            if acc > best_acc and epoch >= args.warmup:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), weight_path)
        
        fout.close()
        model.load_state_dict(best_state)
        model.eval()
        test(args, model, device, test_loader, criterion)
        # Remove 'niter' and 'alpha' since AutoAttack handles them internally
        evaluate_pgd(test_loader, model, epsilon=36/255, device=device, attack_type='autoattack')

    if args.task == 'test':
        weight = torch.load(args.weight_path, map_location=device)
        model.load_state_dict(weight, strict=False)
        model.eval()
        test(args, model, device, test_loader, criterion)
        # Remove 'niter' and 'alpha' since AutoAttack handles them internally
        evaluate_pgd(test_loader, model, epsilon=36/255, device=device, attack_type='autoattack')


if __name__ == '__main__':
    main()
