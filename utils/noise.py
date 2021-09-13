import torch
import numpy as np
import copy

def perturb_eta(eta: torch.tensor, noise_type: str, noise_strength: float):
    eta_tilde = copy.deepcopy(eta)
    if noise_type == 'linear':
        for i in range(len(eta)):
            eta_i = eta[i].squeeze()
            max_ind, sec_ind = eta_i.argsort(descending=True)[:2]
            delta_eta = eta_i[max_ind] - eta_i[sec_ind]
            # Always preserve Bayess optimal prediction (Our assumption)
            eta_i[max_ind] = eta_i[max_ind] - noise_strength*delta_eta
            eta_i[sec_ind] = eta_i[sec_ind] + noise_strength*delta_eta
            eta_tilde[i] = eta_i
    else:
        for i in range(len(eta)):
            eta_i = eta[i].squeeze()
            max_ind = eta_i.argmax()[0]
            dirich_samp = np.random.dirichlet(np.ones(len(eta_i)), 1)
            samp_max_ind = dirich_samp.argmax()[0]
            # Always preserve Bayess optimal prediction (Our assumption)
            eta_i[max_ind], eta_i[samp_max_ind] = eta_i[samp_max_ind], eta_i[max_ind]
            eta_tilde[i] = eta_i
    return eta_tilde