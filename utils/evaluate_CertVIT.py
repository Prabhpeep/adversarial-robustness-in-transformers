from math import ceil
import numpy as np
import torch
import torch.nn as nn 
from tqdm import tqdm 


import torchattacks

# TODO: use args not hard code 

def evaluate_pgd(loader, model, epsilon, niter, alpha, device):
    model.eval()
    correct = 0
    print(epsilon, niter, alpha)

    attack = torchattacks.PGDL2(
        model,
        eps=epsilon,
        alpha=alpha,
        steps=niter,
        random_start=True
    )

    for i, (X, y) in tqdm(enumerate(loader)):
        X, y = X.to(device), y.to(device)

        # --- IMPORTANT: disable parameter grads ---
        for p in model.parameters():
            p.requires_grad_(False)

        X_pgd = attack(X, y)

        # --- re-enable parameter grads ---
        for p in model.parameters():
            p.requires_grad_(True)

        out = model(X_pgd)
        pred = out.argmax(dim=1, keepdim=True)
        correct += pred.eq(y.view_as(pred)).sum().item()

        if i * X.shape[0] > 1000:
            break

    acc = 100. * correct / (i * X.shape[0])
    print(f"PGD Accuracy {acc:.2f}")
    return acc


def vra_sparse(y_true, y_pred):
    labels = y_true[:,0]

    return torch.sum(labels.eq(torch.argmax(y_pred, axis=1)).float())

def vra_cat(y_true, y_pred):
    labels = torch.argmax(y_true, axis=1)[:,None]

    return vra_sparse(labels, y_pred)

def vra(y_true, y_pred):
    if y_true.shape[1] == 1: 
        return vra_sparse(y_true, y_pred)
    else: 
        return vra_cat(y_true, y_pred)
