import torch, IPython
import time
from utils import *

class HessianVectorSolver:
    def __init__(self, model, should_unsqueeze=False):
        self.model = model
        self.unsqueeze = should_unsqueeze

    def apply_Hv(self, grad, v):
        v, grad_vec = torch.squeeze(v), torch.squeeze(grad)
        grad_product = torch.dot(grad_vec, v)
        # take the second gradient
        off = time.time()
        grad_grad = torch.autograd.grad(grad_product, self.model.parameters(), retain_graph=True)
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in grad_grad])
        if self.unsqueeze:
            # add a singleton dimension (e.g. when computing torch.mm with it later)
            hessian_vec_prod = torch.unsqueeze(hessian_vec_prod, 1)
        return hessian_vec_prod.data

def Hv(model, loss, v):
    v = torch.squeeze(v)
    grad_dict = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
    grad_product = torch.dot(grad_vec, v)

    # take the second gradient
    grad_grad = torch.autograd.grad(grad_product, model.parameters())
    # concatenate the results over the different components of the network
    hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in grad_grad])
    return hessian_vec_prod.data


class EigenValue:
    def __init__(self, model):
        self.model = model
        self.hessian_solver = HessianVectorSolver(self.model)

    def largest_eigenvalue(self, grad, max_iter=250, eps=1e-4, max_lam=10.):
        grad = torch.squeeze(grad)
        v = self.normalized_vec_like(grad)
        old_min_lam, old_max_lam = 1E10, 1E10

        # find largest eigenvalue
        for i in range(max_iter):
            v_new = self.hessian_solver.apply_Hv(grad, v)
            if i > 0:
                max_lam = v @ v_new
                if torch.norm(old_max_lam - max_lam) < eps:
                    break
                old_max_lam = max_lam
            v_new /= v_new.norm()
            v = v_new

        large_str = "Largest Eigenvalue: {} after {} iterations".format(max_lam, i)
        return max_lam.data, v

    def largest_smallest_eigenvalue(self, grad, max_iter=250, eps=1e-4, max_lam=10.):
        max_lam, _ = self.largest_eigenvalue(grad, max_iter, eps, max_lam)

        v = self.normalized_vec_like(grad)
        beta = 1. / max_lam
        for i in range(max_iter):
            v_new = v - beta * self.hessian_solver.apply_Hv(grad, v)
            if i > 0:
                min_lam = (1 - v @ v_new) / beta
                if torch.norm(old_min_lam - min_lam) < eps:
                    break
                old_min_lam = min_lam
            v_new /= v_new.norm()
            v = v_new
        print_str = "Smallest Eigenvalue: {} after {} iterations".format(min_lam, i)

        return min_lam.data, max_lam

    def gradient_computation(self, data, loss, model):
        (x,y) = data
        y_pred = model(x)
        loss_val = loss(y_pred, y, model=self.model)
        grad_dict = torch.autograd.grad(loss_val, model.parameters(), create_graph=True)
        grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
        return grad_vec

    def largest_eigenvalue_with_gradient(self, data, loss, model, max_iter=250, eps=1e-4, max_lam=10.):
        grad = self.gradient_computation(data, loss, model)
        max_lam, v_max = self.largest_eigenvalue(grad, max_iter, eps, max_lam)
        return max_lam


    def normalized_vec_like(self, grad):
        v = torch.ones_like(grad).normal_().type(t_FloatTensor)
        return v/v.norm()