from espnet2.tasks.asr import ASRTask
import torch
import logging
from torch import nn
from decimal import Decimal
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Process text files to add punctuation, capitalize sentences, and handle special tokens.")
parser.add_argument("model", type=str, help="Directory containing the text files to process.")
parser.add_argument("--alpha", type=float, default=1.0, help="Weight for the adaptation")
parser.add_argument("--rest_alpha", type=float, default=0.5, help="alpha for the remaining parameters")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load current model
asr_train_config = f"{args.model}/config.yaml"
asr_model_file   = f"{args.model}/valid.acc.ave.pth"
device = "cpu"

asr_model, asr_train_args = ASRTask.build_model_from_file(
        asr_train_config, asr_model_file, device
)

# load previous model
init_model = torch.load(f"{args.model}/initial_model.pth", map_location='cpu')

# load hessian approximations
hessian = torch.load("/".join(asr_train_args.init_param[0].split("/")[:-1]) + "/importance_weight.kf", map_location='cpu')

# to store new model
new_model = {}

all_values = []
weighted_mean = 0
params = 0

def _sym_damp(M: torch.Tensor, damping: float) -> torch.Tensor:
    # Ensure PSD and well-conditioned
    return 0.5 * (M + M.transpose(-1, -2)) + damping * torch.eye(M.size(-1), device=M.device, dtype=M.dtype)

def _chol_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Solve A X = B with Cholesky (A SPD)
    L = torch.linalg.cholesky(A)
    return torch.cholesky_solve(B, L)

def ihvp_kron(
    update: torch.Tensor,
    H_list,
    Q_list,
    damping: float = 1e-3,
    maxiter: int = 100,
    tol: float = 1e-6,
    return_inverse_quadratic: bool = False,
):
    """
    Compute inverse-Hessian vector product X for Kronecker-factored blocks:
        sum_k H_k X Q_k = update
    where update, X are (d_out x d_in).

    Args:
        update: torch.Tensor (d_out, d_in)  -- the 'vector' to multiply by (sum_k H_k ⊗ Q_k^T)^(-1)
        H_list: list[Tensor] or Tensor     -- each H_k is (d_out, d_out)
        Q_list: list[Tensor] or Tensor     -- each Q_k is (d_in, d_in)
        damping: float                      -- Tikhonov damping added to each factor
        maxiter: int                        -- CG iterations for multi-pair case
        tol: float                          -- CG tolerance (Frobenius norm)
        return_inverse_quadratic: bool      -- also return <update, X>

    Returns:
        X: torch.Tensor (d_out, d_in)
        (optional) inv_quad: scalar tensor = <update, X>
    """
    # Normalize inputs to lists
    if isinstance(H_list, torch.Tensor): H_list = [H_list]
    if isinstance(Q_list, torch.Tensor): Q_list = [Q_list]
    assert len(H_list) == len(Q_list), "H_list and Q_list must have same length"

    # Dampen & symmetrize
    Hs = [_sym_damp(H, damping) for H in H_list]
    Qs = [_sym_damp(Q, damping) for Q in Q_list]

    # Fast path: single pair => closed form X = H^{-1} U Q^{-1}
    if len(Hs) == 1:
        H = Hs[0]; Q = Qs[0]
        # Solve H Y = update
        Y = _chol_solve(H, update)
        # Solve X Q = Y  ->  X = Y Q^{-1}
        # Right-side solve via left solve on Q^T: X^T = (Q^T)^{-1} Y^T
        X = _chol_solve(Q.transpose(-1, -2), Y.transpose(-1, -2)).transpose(-1, -2)
        if return_inverse_quadratic:
            inv_quad = torch.sum(update * X)
            return X, inv_quad
        return X

    # Multi-pair: solve A(X)=update with matrix-free CG, where A(X)=sum_k H_k X Q_k
    def A(X):
        Y = torch.zeros_like(X)
        for H, Q in zip(Hs, Qs):
            Y = Y + H @ X @ Q
        return Y

    # Conjugate Gradient in Frobenius inner product
    B = update
    X = torch.zeros_like(B)
    R = B - A(X)
    P = R.clone()
    rTr = torch.sum(R * R)

    for _ in range(maxiter):
        AP = A(P)
        pTAp = torch.sum(P * AP)
        # Guard against division by zero
        if pTAp.abs() < 1e-20:
            break
        alpha = rTr / pTAp
        X = X + alpha * P
        R_new = R - alpha * AP
        rTr_new = torch.sum(R_new * R_new)
        if torch.sqrt(rTr_new) <= tol * (torch.sqrt(torch.sum(B * B)) + 1e-12):
            R = R_new
            rTr = rTr_new
            break
        beta = rTr_new / rTr
        P = R_new + beta * P
        R = R_new
        rTr = rTr_new

    if return_inverse_quadratic:
        inv_quad = torch.sum(update * X)
        return X, inv_quad
    return X


# apply inverse Hessian regularization to linear layers
for name, module in asr_model.named_modules():
    if isinstance(module, torch.nn.Linear):
       # compute difference
       delta_W = module.weight.data.cpu() - init_model[name + ".weight"]

       logging.info(f"Processing layer... {name}")

       gated_update = ihvp_kron(delta_W, hessian[name]['H'], hessian[name]['Q'])
       gated_update = args.alpha * gated_update / torch.norm(gated_update) * torch.norm(delta_W) 

       # update weight in new_model
       new_model[name + ".weight"] = init_model[name + ".weight"] + gated_update


# apply averaging to the rest of the model
for name, param in asr_model.state_dict().items():
    if name in new_model.keys():
        continue
    delta_W = param.cpu() - init_model[name] 

    gated_update = args.rest_alpha * delta_W
    new_model[name] = init_model[name] + gated_update

torch.save(new_model, f"{args.model}/ihr_model.pth")
logging.info(f"Done!")
