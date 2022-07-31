from re import S
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim, autograd
from torch import distributions as dist
from torch.nn.utils.convert_parameters import parameters_to_vector
from tqdm.auto import tqdm, trange
from toolz import itertoolz
import numpy as np

# >>> Utils 
class MaskedLinear(nn.Module):
    
    def __init__(self, base_layer, m_in, m_out):
        """
        The standard nn.Linear layer, but with gradient masking to enforce the LULA construction.
        """
        super(MaskedLinear, self).__init__()

        # Extend the weight matrix
        W_base = base_layer.weight.data.clone()  # (n_out, n_in)
        n_out, n_in = W_base.shape

        W = torch.randn(n_out+m_out, n_in+m_in)
        W[0:n_out, 0:n_in] = W_base.clone()
        W[0:n_out, n_in:] = 0  # Upper-right quadrant

        self.weight = nn.Parameter(W)

        # Extend the bias vector
        if base_layer.bias is not None:
            b_base = base_layer.bias.data.clone()

            b = torch.randn(n_out+m_out)
            b[:n_out] = b_base.clone()

            self.bias = nn.Parameter(b)
        else:
            self.bias = None

        # Apply gradient mask to the weight and bias
        self.mask_w = torch.zeros(n_out+m_out, n_in+m_in)
        self.mask_w[n_out:, :] = 1  # Lower half

        self.mask_b = torch.zeros(n_out+m_out)
        self.mask_b[n_out:] = 1

        self.switch_grad_mask(True)

        # For safekeeping
        self.W_base, self.b_base = W_base, b_base
        self.n_out, self.n_in = n_out, n_in
        self.m_out, self.m_in = m_out, m_in

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def switch_grad_mask(self, on=True):
        if on:
            self.grad_handle_w = self.weight.register_hook(lambda grad: grad.mul_(self.mask_w))
            self.grad_handle_b = self.bias.register_hook(lambda grad: grad.mul_(self.mask_b))
        else:
            self.grad_handle_w.remove()
            self.grad_handle_b.remove()

    def to_gpu(self):
        self.mask_w = self.mask_w.cuda()
        self.mask_b = self.mask_b.cuda()

    def to_unmasked(self):
        lin = nn.Linear(self.n_in+self.m_in, self.n_out+self.m_out)
        lin.weight = self.weight
        lin.bias = self.bias
        return lin

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.n_in+self.m_in, self.n_out+self.m_out, self.bias is not None
        )


class MaskedConv2d(nn.Module):

    def __init__(self, base_layer, m_in, m_out):
        """
        The standard nn.Conv2d layer, but with gradient masking to enforce the LULA construction.
        """
        super(MaskedConv2d, self).__init__()

        self.kernel_size = base_layer.kernel_size
        self.stride  = base_layer.stride
        self.padding = base_layer.padding
        self.dilation = base_layer.dilation
        self.groups = base_layer.groups

        # Extend the weight matrix
        W_base = base_layer.weight.data.clone()  # (n_out, n_in, k, k)
        n_out, n_in, k, _ = W_base.shape  # Num of channels

        W = torch.randn(n_out+m_out, n_in+m_in, k, k)
        W[0:n_out, 0:n_in, :, :] = W_base.clone()
        W[0:n_out, n_in:, :, :] = 0  # Upper-right quadrant

        self.out_channels, self.in_channels = W.shape[0], W.shape[1]
        
        self.weight = nn.Parameter(W)

        # Extend the bias vector
        if base_layer.bias is not None:
            b_base = base_layer.bias.data.clone()

            b = torch.randn(n_out+m_out)
            b[:n_out] = b_base.clone()

            self.bias = nn.Parameter(b)
        else:
            b_base = None
            self.bias = None

        # Apply gradient mask to the weight and bias
        self.mask_w = torch.zeros(n_out+m_out, n_in+m_in, k, k)
        self.mask_w[n_out:, :, :, :] = 1  # Lower half
        self.mask_w = self.mask_w.cuda()
        
        self.mask_b = torch.zeros(n_out+m_out)
        self.mask_b[n_out:] = 1
        self.mask_b = self.mask_b.cuda()

        self.switch_grad_mask(True)

        # For safekeeping
        self.W_base, self.b_base = W_base, b_base
        self.n_out, self.n_in = n_out, n_in
        self.m_out, self.m_in = m_out, m_in

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def switch_grad_mask(self, on=True):
        if on:
            self.grad_handle_w = self.weight.register_hook(lambda grad: grad.mul_(self.mask_w))
            if self.bias:
                self.grad_handle_b = self.bias.register_hook(lambda grad: grad.mul_(self.mask_b))
        else:
            self.grad_handle_w.remove()
            if hasattr(self, 'grad_handle_b'):
                self.grad_handle_b.remove()

    def to_gpu(self):
        self.mask_w = self.mask_w.cuda()
        self.mask_b = self.mask_b.cuda()

    def to_unmasked(self):
        conv = nn.Conv2d(self.n_in+self.m_in, self.n_out+self.m_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups)
        conv.weight = self.weight
        conv.bias = self.bias
        return conv

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, bias={}'.format(
            self.n_in+self.m_in, self.n_out+self.m_out, self.bias is not None
        )


# >>> Model 
class LULA_BasicBlock(nn.Module):
    
    def __init__(self, base_model, n_lula_units):
        """
        n_lula_units: Must be list of natural numbers with length equal to the number of hidden layers.
        """

        super(LULA_BasicBlock, self).__init__()

        # Augment all fc layers
        base_modules = [m for m in base_model.modules()
                        if type(m) != nn.Sequential
                           and type(m) != type(base_model)]

        # Zero unit for both input and output layers
        n_lula_units = [0] + n_lula_units + [0]
        assert len(n_lula_units) == 1 + len([m for m in base_modules if type(m) == nn.Linear or type(m) == nn.Conv2d])

        modules = []
        i = 0
        prev_module = None

        for m in base_modules:
            if type(m) == nn.Linear:
                m_in, m_out = n_lula_units[i], n_lula_units[i+1]
                i += 1
                modules.append(MaskedLinear(m, m_in, m_out))
            elif type(m) == nn.Conv2d:
                m_in, m_out = n_lula_units[i], n_lula_units[i+1]
                i += 1
                modules.append(MaskedConv2d(m, m_in, m_out))
            else:
                modules.append(m)
                
        self.lula_conv1 = modules[0]
        self.bn1  = torch.nn.BatchNorm2d(self.lula_conv1.out_channels)
        self.relu = modules[2]
        self.lula_conv2 = modules[3]
        self.bn2  = torch.nn.BatchNorm2d(self.lula_conv2.out_channels)
        self.downsample = base_model.downsample


    def forward(self, x):
        identity = x
        
        out = self.lula_conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.lula_conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out

    def enable_grad_mask(self):
        for m in self.modules():
            if type(m) == MaskedLinear or type(m) == MaskedConv2d:
                m.switch_grad_mask(True)

    def disable_grad_mask(self):
        for m in self.modules():
            if type(m) == MaskedLinear or type(m) == MaskedConv2d:
                m.switch_grad_mask(False)

    def to_gpu(self):
        for m in self.modules():
            if type(m) == MaskedLinear or type(m) == MaskedConv2d:
                m.to_gpu()


class LULAModel(nn.Module):

    def __init__(self, base_model, n_lula_units):
        """
        base_model: Must have a method called `features(x)` and module called `clf`
        n_lula_units: Must be list of natural numbers with length equal to the number of hidden layers.
        """

        super(LULAModel, self).__init__()

        self.base_model = base_model
        # reset the last conv2d block
        self.lula_layer = LULA_BasicBlock(base_model.module.layer4[1], n_lula_units)
        self.base_model.module.layer4[1] = self.lula_layer

    def forward(self, x):
        return self.base_model(x)

    def features(self, x):
        return self.base_model.module.partial_forward(x)

    def enable_grad_mask(self):
        self.lula_layer.enable_grad_mask()

    def disable_grad_mask(self):
        self.lula_layer.disable_grad_mask()

    def to_gpu(self):
        self.lula_layer.to_gpu()

    def unmask(self):
        self.lula_layer.lula_conv1 = self.lula_layer.lula_conv1.to_unmasked()
        self.lula_layer.lula_conv2 = self.lula_layer.lula_conv2.to_unmasked()
        
        
# >>> Train
def train_lula_layer(lula_model, nll, in_loader, out_loader, prior_prec, l2_penalty=0, lr=1e-1, n_iter=1,
                     fisher_samples=1, alpha=1, beta=1, max_grad_norm=1000, progressbar=True, mc_samples=10):
    # Train only the last-layer
    for m in lula_model.modules():
        if type(m) == MaskedLinear or type(m) == MaskedConv2d:
            for p in m.parameters():
                p.requires_grad = True
        else:
            for p in m.parameters():
                p.requires_grad = False

    opt = optim.Adam(filter(lambda p: p.requires_grad, lula_model.parameters()), lr=lr, weight_decay=0)
    pbar = trange(n_iter) if progressbar else range(n_iter)

    for it in pbar:
        epoch_loss = 0
        n = 0

        for (_, x_in, _, _), (x_out, _) in zip(in_loader, out_loader):
            x_in, x_out = x_in.cuda(), x_out.cuda()

            lula_model.disable_grad_mask()

            ll_module = lula_model.lula_layer
            
            mu_W_conv1 = ll_module.lula_conv1.weight
            mu_b_conv1 = ll_module.lula_conv1.bias
            mu_W_conv2 = ll_module.lula_conv2.weight
            mu_b_conv2 = ll_module.lula_conv2.bias
            
            fisher_diag_W_conv1, fisher_diag_W_conv2 = get_fisher_diag_layer_(lula_model, x_in, nll, fisher_samples, bias=False)
            sigma_W_conv1, sigma_W_conv2 = 1/(fisher_diag_W_conv1 + prior_prec), 1/(fisher_diag_W_conv2 + prior_prec)

            if mu_b_conv1 and mu_b_conv2:
                fisher_diag_b1, fisher_diag_b2 = get_fisher_diag_layer_(lula_model, x_in, nll, fisher_samples, bias=True)
                sigma_b_conv1, sigma_b_conv2 = 1/(fisher_diag_b1 + prior_prec), 1/(fisher_diag_b2 + prior_prec)

            lula_model.enable_grad_mask()

            py_in, py_out = 0, 0

            for s in range(mc_samples):
                # Sample from the posterior
                W_conv1_s = mu_W_conv1 + torch.sqrt(sigma_W_conv1) * torch.randn(*mu_W_conv1.shape, device='cuda')
                ll_module.lula_conv1.weight = torch.nn.Parameter(W_conv1_s)
                if ll_module.lula_conv1.bias:
                    b_conv1_s = mu_b_conv1 + torch.sqrt(sigma_b_conv1) * torch.randn(*mu_b_conv1.shape, device='cuda')
                    ll_module.lula_conv1.bias   = torch.nn.Parameter(b_conv1_s)
                    
                W_conv2_s = mu_W_conv2 + torch.sqrt(sigma_W_conv2) * torch.randn(*mu_W_conv2.shape, device='cuda')
                ll_module.lula_conv2.weight = torch.nn.Parameter(W_conv2_s)
                if ll_module.lula_conv2.bias:
                    b_conv2_s = mu_b_conv2 + torch.sqrt(sigma_b_conv2) * torch.randn(*mu_b_conv2.shape, device='cuda')
                    ll_module.lula_conv2.bias = torch.nn.Parameter(b_conv2_s)
                
                py_in  += torch.softmax(lula_model(x_in), 1)/mc_samples
                py_out += torch.softmax(lula_model(x_out), 1)/mc_samples
                
            loss_in = dist.Categorical(py_in).entropy().mean()
            loss_out = dist.Categorical(py_out).entropy().mean()

            # Min. in-dist uncertainty, max. out-dist uncertainty
            loss = alpha*loss_in - beta*loss_out

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lula_model.parameters(), max_grad_norm)
            opt.step()
            opt.zero_grad()

            ll_module.lula_conv1.weight = mu_W_conv1
            ll_module.lula_conv1.bias = mu_b_conv1
            ll_module.lula_conv2.weight = mu_W_conv2
            ll_module.lula_conv2.bias = mu_b_conv2 
            
            epoch_loss += loss.detach().item()
            n += 1

        if progressbar:
            epoch_loss /= n
            weight_norm = parameters_to_vector(lula_model.parameters()).norm(2).detach().item()
            pbar.set_description(f'Loss: {epoch_loss:.3f}; Weight norm: {weight_norm:.3f}')

    return lula_model


def get_fisher_diag_layer_(model, x, nll, n_samples, lik_prec=1, bias=False):
    
    fisher_diag_conv1 = 0
    fisher_diag_conv2 = 0

    for s in range(n_samples):
        output = model(x).squeeze()

        # Obtain the diagonal-Fisher approximation to the Hessian
        if type(nll) == nn.BCEWithLogitsLoss:
            y = torch.distributions.Bernoulli(logits=output).sample()
        elif type(nll) == nn.CrossEntropyLoss:
            y = torch.distributions.Categorical(logits=output).sample()
        else:
            y = torch.distributions.Normal(output, lik_prec).sample()

        loss = nll(output, y)

        # TODO: modify following line
        ll_params = model.lula_layer
        p_conv1 = ll_params.lula_conv1.bias if bias else ll_params.lula_conv1.weight
        p_conv2 = ll_params.lula_conv2.bias if bias else ll_params.lula_conv2.weight

        grad_conv1 = autograd.grad([loss], p_conv1, retain_graph=True, create_graph=True)[0]
        grad_conv2 = autograd.grad([loss], p_conv2, retain_graph=True, create_graph=True)[0]
        
        fisher_diag_conv1 += x.shape[0] * grad_conv1**2
        fisher_diag_conv2 += x.shape[0] * grad_conv2**2

    fisher_diag_conv1 /= n_samples
    fisher_diag_conv2 /= n_samples

    return fisher_diag_conv1, fisher_diag_conv2


@torch.no_grad()
def kfla_predict(test_loader, model, n_samples=20, apply_softmax=True, return_targets=False, delta=1, n_data=None):
    py = []
    targets = []
    count = 0

    for batch in test_loader:
        if len(batch)==4:
            _, x, y, _ = batch 
        else:
            x, y = batch
        
        if n_data is not None and count >= n_data:
            break

        x, y = delta*x.cuda(), y.cuda()
        targets.append(y)

        # MC-integral
        py_ = 0
        for _ in range(n_samples):
            out = model.forward_sample(x)
            py_ += torch.softmax(out, 1) if apply_softmax else out

        py_ /= n_samples
        py.append(py_)
        count += len(x)

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets, dim=0)
    else:
        return torch.cat(py, dim=0)


@torch.no_grad()
def lula_predict(dataloader, model, n_samples=1, T=1, delta=1, return_targets=False):
    py = []
    targets = []

    for _, x, y, _ in dataloader:
        x = delta*x.cuda()

        py_ = 0
        for _ in range(n_samples):
            f_s = model.forward(x)
            py_ += torch.softmax(f_s/T, 1)
        py_ /= n_samples

        py.append(py_)
        targets.append(y)

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets, dim=0)
    else:
        return torch.cat(py, dim=0)


class KFAC(optim.Optimizer):
    
    def __init__(self, net, alpha=0.95):
        """ K-FAC Preconditionner for Linear and Conv2d layers.
        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.
        Args:
            net (torch.nn.Module): Network to precondition.
            alpha (float): Running average parameter (if == 1, no r. ave.).
        """
        self.alpha = alpha
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0

        mod = net.base_model.module.fc
        handle = mod.register_forward_pre_hook(self._save_input)
        self._fwd_handles.append(handle)

        handle = mod.register_backward_hook(self._save_grad_output)
        self._bwd_handles.append(handle)

        params = [mod.weight]

        if mod.bias is not None:
            params.append(mod.bias)

        d = {'params': params, 'mod': mod, 'layer_type': mod.__class__.__name__}
        self.params.append(d)

        super(KFAC, self).__init__(self.params, {})

    def step(self):
        for group in self.param_groups:
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None

            state = self.state[group['mod']]
            self._compute_covs(group, state)

        self._iteration_counter += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        mod = group['mod']
        x  = state['x']
        gy = state['gy']

        # Computation of xxt
        if group['layer_type'] == 'Conv2d' or group['layer_type'] == 'MaskedConv2d':
        # if group['layer_type'] == 'Conv2d':
            x = F.unfold(x, mod.kernel_size, padding=mod.padding,
                             stride=mod.stride)
            x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()

        # if mod.bias is not None:
        #     ones = torch.ones_like(x[:1])
        #     x = torch.cat([x, ones], dim=0)

        if self._iteration_counter == 0:
            state['xxt'] = torch.mm(x, x.t()) / float(x.shape[1])
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(x.shape[1]))

        # Computation of ggt
        if group['layer_type'] == 'Conv2d' or group['layer_type'] == 'MaskedConv2d':
        # if group['layer_type'] == 'Conv2d':
            gy = gy.data.permute(1, 0, 2, 3)
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state['num_locations'] = 1

        if self._iteration_counter == 0:
            state['ggt'] = torch.mm(gy, gy.t()) / float(gy.shape[1])
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(gy.shape[1]))


class KFLA(nn.Module):
    """
    Taken, with modification, from:
    https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/diag_laplace.py
    In particular, this is KFLA is only for linear layers
    """

    def __init__(self, base_model):
        super().__init__()

        self.net = base_model
        self.params = []
        self.net.apply(lambda module: kfla_parameters(module, self.params))
        self.hessians = None

    def forward(self, x):
        return self.net.forward(x)

    def forward_sample(self, x):
        self.sample()
        return self.net.forward(x)

    def sample(self, scale=1, require_grad=False):
        for module, name in self.params:
            mod_class = module.__class__.__name__

            if mod_class not in ['Linear', 'MaskedLinear']:
                continue

            if name == 'bias':
                w = module.__getattr__(f'{name}_mean')
            else:
                M = module.__getattr__(f'{name}_mean')
                U_half = module.__getattr__(f'{name}_U_half')
                V_half = module.__getattr__(f'{name}_V_half')

                if len(M.shape) == 1:
                    M_ = M.unsqueeze(1)
                elif len(M.shape) > 2:
                    M_ = M.reshape(M.shape[0], np.prod(M.shape[1:]))
                else:
                    M_ = M

                E = torch.randn(*M_.shape, device='cuda')
                w = M_ + scale * U_half @ E @ V_half
                w = w.reshape(*M.shape)

            if require_grad:
                w.requires_grad_()

            module.__setattr__(name, w)

    def estimate_variance(self, var0, invert=True):

        tau = 1/var0
        U, V = self.hessians

        for module, name in self.params:
            
            if module.in_features == 513:
                continue
            
            if name == 'bias':
                continue
            
            if module.__class__.__name__ not in ['Linear', 'MaskedLinear']:
                continue
            
            U_ = U[(module, name)].clone()
            V_ = V[(module, name)].clone()

            if invert:
                m, n = int(U_.shape[0]), int(V_.shape[0])

                U_ += torch.sqrt(tau)*torch.eye(m, device='cuda')
                V_ += torch.sqrt(tau)*torch.eye(n, device='cuda')

                # For numerical stability
                u = torch.cholesky(U_.cpu() + 1e-6*torch.eye(m))
                v = torch.cholesky(V_.cpu() + 1e-6*torch.eye(n))

                U_ = torch.cholesky(torch.cholesky_inverse(u), upper=False).cuda()
                V_ = torch.cholesky(torch.cholesky_inverse(v), upper=True).cuda()

            module.__getattr__(f'{name}_U_half').copy_(U_)
            module.__getattr__(f'{name}_V_half').copy_(V_)


    def get_hessian(self, train_loader, binary=False):
        criterion = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
        opt = KFAC(self.net)
        U = {}
        V = {}

        # Populate parameters with the means
        self.sample(scale=0, require_grad=True)

        for _, x, y, _ in tqdm(train_loader, ncols=100):
            x = x.cuda(non_blocking=True)

            self.net.zero_grad()
            out = self(x).squeeze()

            if binary:
                distribution = torch.distributions.Binomial(logits=out)
            else:
                distribution = torch.distributions.Categorical(logits=out)

            y = distribution.sample()
            loss = criterion(out, y)
            loss.backward()
            opt.step()

        with torch.no_grad():
            for group in opt.param_groups:
                if len(group['params']) == 2:
                    weight, bias = group['params']
                else:
                    weight = group['params'][0]
                    bias = None

                module = group['mod']
                state = opt.state[module]

                U_ = state['ggt']
                V_ = state['xxt']

                n_data = len(train_loader.dataset)

                U[(module, 'weight')] = np.sqrt(n_data)*U_
                V[(module, 'weight')] = np.sqrt(n_data)*V_

            self.hessians = (U, V)


    def gridsearch_var0(self, val_loader, ood_loader, interval, n_classes=10, lam=1):
        vals, var0s = [], []
        pbar = tqdm(interval, ncols=100)

        for var0 in pbar:
            try:
                self.estimate_variance(var0)

                preds_in, y_in = kfla_predict(val_loader, self, n_samples=5, return_targets=True)
                loss_in = F.nll_loss(torch.log(preds_in + 1e-8), y_in)

                if ood_loader is not None:
                    preds_out = kfla_predict(ood_loader, self, n_samples=5)
                    loss_out = -torch.log(preds_out + 1e-8).mean()
                else:
                    loss_out = 0

                loss = loss_in + lam * loss_out
            except RuntimeError:
                loss = float('inf')

            vals.append(loss)
            var0s.append(var0)

            pbar.set_description(f'var0: {var0:.5f}, L-in: {loss_in:.3f}, L-out: {loss_out:.3f}, L: {loss:.3f}')

        best_var0 = var0s[np.argmin(vals)]

        return best_var0


def kfla_parameters(module, params):
    mod_class = module.__class__.__name__
    if mod_class not in ['Linear', 'MaskedLinear']:
        return

    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            continue

        data = module._parameters[name].data
        m, n = int(data.shape[0]), int(np.prod(data.shape[1:]))
        module._parameters.pop(name)
        module.register_buffer(f'{name}_mean', data)
        module.register_buffer(f'{name}_U_half', torch.zeros([m, m], device='cuda'))
        module.register_buffer(f'{name}_V_half', torch.zeros([n, n], device='cuda'))
        module.register_buffer(name, data.new(data.size()).zero_())

        params.append((module, name))