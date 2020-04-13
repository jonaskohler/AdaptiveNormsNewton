
from utils import *
from tqdm import tqdm
from models import MNISTAutoencoder


class StandardOptimization():
    def __init__(self, model, sampler, loss, opt):
        self.sampler = sampler
        self.opt = opt
        self.model = model
        self.loss = loss
        self.stats = StatisticsTracker()
        self.optimizer = self.get_optimizer(self.opt["method"])
        self.iteration = 0

    def __call__(self, max_iterations=100):
        stats = self.stats
        stats.reset_time()

        self.test_model(batch_size=self.opt["test_batch_size"])
        loss_full, grad_full, full_train_acc = self.full_loss_computation()

        iterator = tqdm(range(self.iteration, self.iteration + max_iterations))
        for i in iterator:
            # draw a new batch
            (x, y) = self.sampler(sample_size=self.opt["sample_size"])

            # forward pass, loss & gradient computation
            self.model.train()
            self.optimizer.zero_grad()
            if isinstance(self.model, MNISTAutoencoder):
                x_pred=self.model(x)
                loss= self.loss(x_pred,x)
                train_acc = 0

            else:    
                y_pred = self.model(x)
                loss = self.loss(y_pred, y, l2_alpha=self.opt["l2_alpha"], model=self.model)
                train_acc = self.train_accuracy(y_pred, y)

            loss.backward()

            prev_params = flat_params(self.model)
            self.optimizer.step()

            step_norm = torch.norm(flat_params(self.model) - prev_params)
            gradient_norm = self.get_gradient_norm()


            """---------------- test method on test data -------------------------"""
            if self.condition_true(self.opt["test_every"]):
                self.test_model(batch_size=self.opt["test_batch_size"])

            """---------------- compute loss on full training data ---------------"""
            if self.condition_true(self.opt["full_loss_every_n_statistics"],
                                   counter_based_on=self.opt["statistics_every"]) and i>0:
                loss_full, grad_full, full_train_acc = self.full_loss_computation()
                
            """---------------- store the collected statistics -------------------"""
            if self.condition_true(self.opt["statistics_every"]):
                stats(iteration=self.iteration,
                      batch_loss=loss,
                      batch_gradient_norm=gradient_norm,
                      step_norm=step_norm,
                      full_loss=loss_full,
                      full_gradient_norm=grad_full,
                      sample_size=self.opt["sample_size"],
                      train_acc = train_acc,
                      full_train_acc=full_train_acc)

            # if gradient norm within tolerance, stop optimizing

            if gradient_norm < self.opt["grad_tol"]:
                iterator.close()
                print("Converged!")
                break

            if stats.tracked_time() > self.opt["max_time"]:
                iterator.close()
                print_c("Timeout!", "red")
                break

            self.iteration += 1

        stats.new_time_offset()

        return stats

    def train_accuracy(self, Y_pred, target):
        return torch.mean(torch.eq(target, torch.argmax(Y_pred, 1)).type(t_FloatTensor)).data


    def test_model(self, batch_size="full"):
        if self.sampler.X_test is None: return
        self.stats.pause_timer()
        def accuracy(Y_pred, target):
            return torch.mean(torch.eq(target, torch.argmax(Y_pred, 1)).type(t_FloatTensor))

        X, Y = self.sampler(batch_size, train=False)
        # forward pass
        self.model.eval()
        if isinstance(self.model, MNISTAutoencoder):
            X_pred=self.model(X)
            loss= self.loss(X_pred,X)
            accuracy=0
        else:    
            Y_pred = self.model(X)
            loss = self.loss(Y_pred, Y, l2_alpha=self.opt["l2_alpha"], model=self.model)
            accuracy = accuracy(Y_pred, Y)

        self.stats(iteration=self.iteration, test_loss=loss, test_accuracy=accuracy)
        self.stats.restart_timer()

    def get_optimizer(self, method):
        if method == "SGD":
            return torch.optim.SGD(self.model.parameters(), lr = self.opt["learning_rate"])
        elif method == "Adagrad":
            return torch.optim.Adagrad(self.model.parameters(), lr = self.opt["learning_rate"])
        elif method == "Rmsprop":
        	return torch.optim.RMSprop(self.model.parameters(), lr = self.opt["learning_rate"])
            
        else:
            raise NotImplementedError("Optimizer with name {} not implemented yet.".format(method))


    def get_gradient_norm(self):
        return torch.norm(torch.cat([p.grad.view(-1) for p in self.model.parameters()]))


    def full_loss_computation(self, compute_gradient=True, pause_timer=True):
        if pause_timer:
            self.stats.pause_timer()

        # Sample full data and do FW pass
        (X, Y) = self.sampler(sample_size="full")
        if isinstance(self.model, MNISTAutoencoder):
            X_pred_full=self.model(X)
            loss_full= self.loss(X_pred_full,X)
            full_train_acc=0
        else:    

            y_pred_full = self.model(X)
            loss_full = self.loss(y_pred_full, Y, l2_alpha=self.opt["l2_alpha"], model=self.model)
            full_train_acc = self.train_accuracy(y_pred_full, Y)

        if compute_gradient:
            # Compute the gradient norm
            grad_full = torch.norm(flat_gradient(loss_full, self.model.parameters(), create=False)).item()
        else:
            grad_full = None

        if pause_timer:
            # restart the timer
            self.stats.restart_timer()
        return loss_full.item(), grad_full, full_train_acc

    def condition_true(self, metric, counter_based_on=None):
        if not metric:
            return False

        if counter_based_on is None:
            counter = metric
        else:
            counter = metric * counter_based_on
        return (metric is not None and self.iteration % counter == 0)




