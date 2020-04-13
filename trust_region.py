###########################
### Trust Region Method ###
###########################

import time
from tqdm import tqdm
import torch
import torch.nn as nn
import IPython
from hessian_vector import HessianVectorSolver
from stcg import truncatedCG
from models import MNISTAutoencoder

from utils import *
from gltr import GLTR

class Ellipsoidal_TR:
    def __init__(self, model, sampler, loss, opt):
        """
        :param model: Model that is used.
        :param X: Tensor of Datapoints.
        :param Y: Tensor of target labels.
        :param loss: Some torch.nn loss function.
        :param opt: Dictionary of options for the trust region algorithm.
        :param verbose: Int level of verbosity.
        :param test_dset: (x_test, y_test) if test statistics should be computed.
        """
        self.model = model
        self.loss = loss
        self.d = num_parameters(self.model)
        self.opt = opt
        self.sampler = sampler
        self.stats = StatisticsTracker()
        self.max_krylov_dim = self.opt.get("max_krylov_dim", int(2*self.d))
        self.iteration = 0
        self.gradient_norm = 0
        self.new_epoch = False

    def update_parameters(self, step):
        step_reshaped = reshape_like_layers(step.data, self.model)
        assign_add(self.model, step_reshaped)
        return step_reshaped


    def update_procedure_2nd_order(self, step, batch_data, loss, model_improvement):
        # Update the parameters
        step_reshaped = self.update_parameters(step)

        (x,y) = batch_data

        # compute new loss with a new forward pass and compute the function improvement on the mini-batch
        if isinstance(self.model, MNISTAutoencoder):
            x_pred_new=self.model(x)
            loss_new= self.loss(x_pred_new,x)
        else:
            y_pred_new = self.model(x)
            loss_new = self.loss(y_pred_new, y, l2_alpha=self.opt["l2_alpha"], model=self.model)
        fn_improvement = loss.item() - loss_new.item()

        # If function or model improvement are too small we set the ratio to 1 because of possible numerical instabilities4
        if (abs(fn_improvement) > 1E-15 or abs(model_improvement) > 1E-15) and not abs(model_improvement) < 1E-15:
            rho = fn_improvement / model_improvement
        else:
            model_improvement = 1E-15  # model_imp may not be 0 for possible division in line 183
            rho = 1

        # step quality for the current iterate
        step_quality = self.step_quality(rho)
        success = step_quality is not "unsuccessful"

        if step_quality == "very_successful":
            # increase the tr radius
            self.opt["tr_radius"] = min(self.opt["tr_radius"] * self.opt["increase_mult"],
                                        self.opt["max_tr_radius"])
        elif step_quality == "unsuccessful":
            assign_sub(self.model, step_reshaped)
            self.opt["tr_radius"] /= self.opt["decrease_mult"]

        if success:
            # update the pivot statistics with the info from the current (successful) iterate
            self.pivot_stats.successful_steps_count += 1
            self.pivot_stats.rho_sum += rho
            self.pivot_stats.step_sum = [self.pivot_stats.step_sum[i] + layer_update for i, layer_update in enumerate(
                step_reshaped)] if self.pivot_stats.step_sum is not None else step_reshaped
            self.pivot_stats.model_decrease_sum += model_improvement

        if self.has_reached_pivot_point():
            self.execute_pivot_logic()

        return success

    def execute_pivot_logic(self):
        """
        If we reach a pivot point (only for 2nd order TR), we do the following:
                - Check if we've made progress on the full function compared to the last pivot point
                - if yes: update the pivot (full) loss
                - otherwise: reject all the steps within the period, increase the sample size
        """
        loss_full_new, grad_full_new, full_train_acc = self.full_loss_computation(compute_gradient=True, pause_timer="gradient")
        fn_improvement = self.pivot_stats.last_full_loss - loss_full_new

        if fn_improvement > 0.:
            self.pivot_stats.last_full_loss = loss_full_new
            self.pivot_stats.last_full_gradient = grad_full_new
            self.pivot_stats.last_full_train_acc = full_train_acc
        else:
            assign_sub(self.model, self.pivot_stats.step_sum)
            self.opt["sample_size"] = min(self.opt["sample_size"] * self.opt["batch_increase_mult"], self.opt["batch_max_size"] * self.opt["full_samples"])

        # reset all the values stored in pivot_stats
        self.pivot_stats.reset(self.iteration)


    def solve_2ndorder_subproblem(self, gradients, success):
        skip_loop = False
        # Init Hessian vector solver
        hv_solver = HessianVectorSolver(self.model, should_unsqueeze=False)

        # get the preconditioning matrix
        M = self.preconditioner(gradients.data, update=(success or self.iteration == 0))

        # Solve subproblem
        step, model_improvement, subproblem_info = self.subproblem_solver(gradients,
                                                                     hv_solver.apply_Hv,
                                                                     self.opt["krylov_tol"],
                                                                     self.opt["tr_radius"],
                                                                     M,
                                                                     self.sampler.train_epoch,
                                                                     epochs_1storder=self.opt["1st_order_epochs"],
                                                                     exact_tol=self.opt["exact_tol"],
                                                                     successful_flag=success,
                                                                     exact_solver=self.opt["exact_solver"],
                                                                     max_krylov_dim=self.max_krylov_dim,
                                                                     precon_stopping_criterion=self.opt.get(
                                                                         "precon_stopping_criterion", False),
                                                                     verbose=0)

        if isnan(model_improvement):
            print_c("Model improvement is NaN", "red")
            skip_loop = True
        assert model_improvement >= -1e-12, "Something went wrong. Model improvement negative: {}".format(
            model_improvement)

        return step, model_improvement, subproblem_info, skip_loop



    def __call__(self, max_iterations=100, *args, **kwargs):
        """
        :param max_iterations: Maximum number of optimization iterations.
        """
        start_time = time.time()
        stats = self.stats
        self.pivot_stats = PivotStatistics()

        # Choose subproblem solver
        self.subproblem_solver = self._get_subproblem_solver()

        # Specifiy trust region shape
        self.preconditioner = self.preconditioning_matrix(type=self.opt["preconditioner"], eps=self.opt["epsilon"])

        ## Compute initial test settatistics (test loss and accuracy) and full training loss
        self.test_model(batch_size="full")
        loss_full, grad_full, full_train_acc = self.full_loss_computation()
        self.pivot_stats.last_full_loss = loss_full

        self.optimizer = SGDOptimizer(self.model, self.opt["learning_rate"], type=self.opt["1st_order_optimizer"])

        iterator = tqdm(range(self.iteration, self.iteration + max_iterations))
        stats.reset_time()

        for i in iterator:

            min_e, max_e, rho, fn_improvement, model_improvement = None, None, None, None, None
            success = True

            epoch_count_before = self.sampler.train_epoch
            do_1st_order_step = (self.sampler.train_epoch <= self.opt["1st_order_epochs"])
            sample_size = self.opt["sample_size"] if not do_1st_order_step else self.opt["sample_size_sgd"]
            # draw a new batch
            (x, y) = self.sampler(sample_size=sample_size)
            self.new_epoch = (self.sampler.train_epoch != epoch_count_before)

            # forward pass, loss & gradient computation
            self.model.train()
            if isinstance(self.model, MNISTAutoencoder):
                x_pred=self.model(x)
                loss= self.loss(x_pred,x)
                train_acc=0
            else:
                if self.opt.get("full_function_based_update", False):
                    (x_full, y_full) = self.sampler(sample_size="full")
                    y_pred_full_fn = self.model(x_full)
                    loss_full_fn = self.loss(y_pred_full_fn, y_full, l2_alpha=self.opt["l2_alpha"], model=self.model).item()
                y_pred = self.model(x)
                loss = self.loss(y_pred, y, l2_alpha=self.opt["l2_alpha"], model=self.model)
                train_acc = self.train_accuracy(y_pred, y)

            # compute the subsampled gradients. The create=True flag makes sure we can later backprop through them to obtain the Hessian.
            gradients = flat_gradient(loss, self.model.parameters(), create=(not do_1st_order_step))


            if do_1st_order_step:
                """---------------- SGD STEP -------------------------"""
                step = self.optimizer.step(gradients)
                subproblem_info = {"steps": -1, "info": None}
            else:
                """---------------- 2nd order step -------------------------"""
                step, model_improvement, subproblem_info, skip = self.solve_2ndorder_subproblem(gradients, success)
                if skip: continue
                if self.opt.get("full_function_based_update", False):
                    (x, y) = self.sampler(sample_size="full")
                    success = self.update_procedure_2nd_order(step, (x_full, y_full), loss_full_fn, model_improvement)
                else:
                    success = self.update_procedure_2nd_order(step, (x, y), loss, model_improvement)


            """---------------- test method on test data -------------------------"""
            if self.condition_true(self.opt["test_every"]):
                self.test_model(batch_size="full")

            """---------------- compute loss on full training data ---------------"""
            if self.condition_true(self.opt["full_loss_every_n_statistics"],
                                   counter_based_on=self.opt[
                                       "statistics_every"]) and self.iteration > 0 and (do_1st_order_step or self.opt["pivot_k"] >= 1E10):
                loss_full, grad_full, full_train_acc = self.full_loss_computation()
                self.pivot_stats.last_full_loss = loss_full
                self.pivot_stats.last_full_gradient = grad_full
                self.pivot_stats.last_full_train_acc = full_train_acc
            elif self.condition_true(self.opt["full_loss_every_n_statistics"],
                                     counter_based_on=self.opt["statistics_every"]) and not do_1st_order_step:
                loss_full, grad_full, full_train_acc = self.pivot_stats.last_full_loss, self.pivot_stats.last_full_gradient, self.pivot_stats.last_full_train_acc

            elif not self.condition_true(self.opt["full_loss_every_n_statistics"],
                                     counter_based_on=self.opt["statistics_every"]):
                loss_full, grad_full = None, None

            """---------------- store the collected statistics -------------------"""
            if self.condition_true(self.opt["statistics_every"]):
                stats(iteration=self.iteration,
                      batch_loss=loss,
                      batch_gradient_norm=torch.norm(gradients),
                      step_norm=torch.norm(torch.squeeze(step)),
                      subproblem_info=subproblem_info["info"],
                      sub_steps=subproblem_info["steps"],
                      rho=rho,
                      fn_improvement=fn_improvement,
                      tr_radius=self.opt["tr_radius"],
                      full_loss=loss_full,
                      full_gradient_norm=grad_full,
                      minimum_eigenvalue=min_e,
                      maximum_eigenvalue=max_e,
                      sample_size=sample_size,
                      train_acc=train_acc,
                      full_train_acc = full_train_acc)


            if stats.tracked_time() > self.opt["max_time"]:
                iterator.close()
                print_c("Timeout!", "red")
                break

            self.iteration += 1

        stats.new_time_offset()
        return stats



    def test_model(self, batch_size="full"):
        if self.sampler.X_test is None: return
        self.stats.pause_timer()
        def accuracy(Y_pred, target):
            return torch.mean(torch.eq(target, torch.argmax(Y_pred, 1)).type(t_FloatTensor))

        X, Y = self.sampler(batch_size, train=False)
        # forward pass
        self.model.eval()
        if isinstance(self.model, MNISTAutoencoder):
            x_pred=self.model(X)
            loss= self.loss(x_pred,X)
            accuracy=0
        else:
            Y_pred = self.model(X)
            loss = self.loss(Y_pred, Y, l2_alpha=self.opt["l2_alpha"], model=self.model)
            accuracy = accuracy(Y_pred, Y)

        self.stats(iteration=self.iteration, test_loss=loss, test_accuracy=accuracy)
        self.stats.restart_timer()


    def train_accuracy(self, Y_pred, target):
        return torch.mean(torch.eq(target, torch.argmax(Y_pred, 1)).type(t_FloatTensor)).data


    def step_quality(self, rho):
        if rho >= self.opt["eta_2"]:
            return "very_successful"
        elif rho >= self.opt["eta_1"]:
            return "successful"
        else:
            return "unsuccessful"

    def has_reached_pivot_point(self):
        return self.new_epoch and self.sampler.train_epoch % self.opt["pivot_k"] == 0


    def _get_subproblem_solver(self):
        subproblem_solver = self.opt.get('subproblem_solver', 'CG')

        if subproblem_solver == 'CG':
            subproblem_solver = truncatedCG
        elif subproblem_solver == 'GLTR':
            subproblem_solver = GLTR
        else:
            raise NotImplementedError('Subproblem solver "' + subproblem_solver + '" unknown.')

        return subproblem_solver


    def preconditioning_matrix(self, type, eps=1E-10):
        if type == "uniform":
            return UniformPreconditioning()
        elif type == "adagrad":
            return AdagradPreconditioning(epsilon=eps)
        elif type == "rms":
            return RMSPreconditioning(epsilon=eps)
        elif type=="svag":
            return SVAGPreconditioning()
        else:
            return NotImplementedError()


    def condition_true(self, metric, counter_based_on=None):
        if not metric:
            return False

        if counter_based_on is None:
            counter = metric
        else:
            counter = metric * counter_based_on
        return (metric is not None and self.iteration % counter == 0)


    def full_loss_computation(self, compute_gradient=True, pause_timer="both"):
        """
        :param compute_gradient:
        :param pause_timer: can be "both", "gradient", "none"
        :return:
        """
        paused_timer = False
        if pause_timer == "both":
            self.stats.pause_timer()
            paused_timer = True

        # Sample full data and do FW pass
        (X, Y) = self.sampler(sample_size="full")
        if isinstance(self.model, MNISTAutoencoder):
            x_pred_full=self.model(X)
            loss_full= self.loss(x_pred_full,X)
            full_train_acc=0
        else:
            y_pred_full = self.model(X)
            loss_full = self.loss(y_pred_full, Y, l2_alpha=self.opt["l2_alpha"], model=self.model)
            full_train_acc = self.train_accuracy(y_pred_full, Y)

        if compute_gradient:
            if pause_timer == "gradient" and not paused_timer:
                self.stats.pause_timer()
                paused_timer = True
            # Compute the gradient norm
            grad_full = torch.norm(flat_gradient(loss_full, self.model.parameters(), create=False)).item()
        else:
            grad_full = None

        if paused_timer:
            # restart the timer
            self.stats.restart_timer()
            paused_timer = False

        return loss_full.item(), grad_full, full_train_acc

    
class PivotStatistics:
    def __init__(self):
        self.successful_steps_count = 0.
        self.rho_sum = 0.
        self.step_sum = None
        self.model_decrease_sum = 0.
        self.last_full_loss = 1e10
        self.last_pivot_iteration = 0
        self.last_full_gradient = 0
        self.last_full_train_acc = 0

    def reset(self, iteration):
        self.successful_steps_count = 0.
        self.rho_sum = 0.
        self.step_sum = None
        self.model_decrease_sum = 0.
        self.last_pivot_iteration = iteration


class SGDOptimizer:
    def __init__(self, model,learning_rate, type="SGD", eps=1e-10):
        if type == "SGD":
            self.preconditioning = UniformPreconditioning()
        elif type == "Adagrad":
            self.preconditioning = AdagradPreconditioning(epsilon=eps)
        elif type == "Rmsprop":
            self.preconditioning = RMSPreconditioning(epsilon=eps)
        else:
            raise NotImplementedError
        self.learning_rate = learning_rate
        self.model = model

    def step(self, gradients):
        # if necessary flatten the gradient
        if len(gradients.size()) > 1:
            gradients = torch.cat([g.contiguous().view(-1) for g in gradients])

        update = - self.learning_rate * 1/self.preconditioning(gradients) * gradients
        update_reshaped = reshape_like_layers(update, self.model)
        assign_add(self.model, update_reshaped)
        return update


class UniformPreconditioning:
    def __call__(self, gradient, **kwargs):
        return torch.ones_like(gradient)


class AdagradPreconditioning:
    def __init__(self, epsilon=1e-8): ###this is changed in grid_search now.

        self.g = None
        self.epsilon = epsilon

    def __call__(self, gradient, update=True):
        if self.g is None:
            self.g = torch.ones_like(gradient) * self.epsilon

        if update:
            self.g += gradient.clone().detach()**2
        return torch.sqrt(self.g)

class RMSPreconditioning:
    def __init__(self, epsilon=1e-10, beta=0.8):

        self.g = None
        self.epsilon = epsilon
        self.beta = beta

    def __call__(self, gradient, update=True):
        if self.g is None:
            self.g = torch.ones_like(gradient) * self.epsilon

        if update:
            self.g = self.beta * self.g + (1-self.beta) * gradient.clone().detach()**2
        return torch.sqrt(self.g )

class SVAGPreconditioning:
    def __init__(self, epsilon=1e-18, beta=0.9):
        self.v = None
        self.m= None
        self.factor=None

        self.epsilon = epsilon
        self.beta = beta

        self.t=0

    def __call__(self, gradient, update=True):
        if self.v is None:
            self.v = torch.zeros_like(gradient) * self.epsilon

        if self.m is None:
            self.m=  torch.zeros_like(gradient)*self.epsilon
        if self.factor is None:
            self.factor= torch.zeros_like(gradient)

        if update:
            self.m= self.beta*self.m + + (1-self.beta) * gradient.clone().detach()
            self.v = self.beta *self.v + (1-self.beta) * gradient.clone().detach()**2
            m=self.m/(1-self.beta**(self.t+1))
            m_sq=m**2
            v=self.v/(1-self.beta**(self.t+1))

            bias=((1-self.beta)*(1+self.beta**(self.t+1)))/((1+self.beta)*(1-self.beta**(self.t+1)))
            s=(v-m_sq)/(1-bias)+self.epsilon
            self.fact=s/m_sq
            self.t+=1
    
        return torch.ones_like(gradient) + self.factor





