import torch
import time
import numpy as np, pandas as pd
import IPython
from tensorboardX import SummaryWriter
import os, json
from termcolor import colored

def use_gpu():
    return torch.cuda.device_count() > 0

if use_gpu():
    t_FloatTensor = torch.cuda.FloatTensor
    t_DoubleTensor = torch.cuda.DoubleTensor
    t_IntTensor = torch.cuda.IntTensor
    t_LongTensor = torch.cuda.LongTensor
else:
    t_FloatTensor = torch.FloatTensor
    t_DoubleTensor = torch.DoubleTensor
    t_IntTensor = torch.IntTensor
    t_LongTensor = torch.LongTensor

def make_path(path):
    dirs = path.split("/")
    dirs = ["/".join(dirs[:i+1]) for i in range(len(dirs))]
    for _dir in dirs:
        if not os.path.isdir(_dir):
            os.makedirs(_dir)

def print_c(txt, color="white"):
    print(colored(txt, color))

def flatten_parameters(model):
    flattened_data, flattened_grad = [], []
    for p in model.parameters():
        flattened_data.append(p.data.view(-1))
        flattened_grad.append(p.grad.view(-1))
    return torch.cat(flattened_data), torch.cat(flattened_grad)

def flat_params(model):
    flat_data = []
    for p in model.parameters():
        flat_data.append(p.data.view(-1))
    return torch.cat(flat_data)

def num_parameters(model):
    return sum(p.numel() for p in model.parameters())

def almost_equal(tensor1, tensor2, eps=1E-4):
    return bool(torch.all(torch.lt(torch.abs(tensor1 - tensor2), eps)))

def flat_gradient(loss, params, create=True):
    grad_dict = torch.autograd.grad(loss, params, create_graph=create)
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
    return grad_vec

def reshape_like_layers(vec, model):
    vec = vec.squeeze()
    vec_list = []
    offset = 0
    for p in model.parameters():
        span = int(torch.prod(torch.Tensor(list(p.data.size()))))
        start, end = offset, offset + span
        vec_list.append(vec[start:end].view(p.size()))
        offset += span
    return vec_list

def assign_sub(model, step):
    for i, p in enumerate(model.parameters()):
        p.data -= step[i].data

def assign_add(model, step):
    for i, p in enumerate(model.parameters()):
        p.data += step[i].data


def isnan(x):
    return x != x

def copy_of_modelweights(model):
    return [p.data.clone() for p in model.parameters()]

def get_travelled_distance(model, init_point):
    init_w = torch.cat([val.view(-1) for val in init_point])
    current_w = torch.cat([p.data.view(-1) for p in model.parameters()])
    return torch.norm(init_w-current_w)

def copy_dict_with_changes(dic_to_copy: dict, **kwargs):
    dict = dic_to_copy.copy()
    for key, value in kwargs.items():
        dict[key] = value
    return dict

def copy_dict_with_changes_from_dict(dic_to_copy: dict, new_dic: dict):
    dict = dic_to_copy.copy()
    if new_dic is None: return dict
    for key, value in new_dic.items():
        dict[key] = value
    return dict


def new_run_directory(_dir):
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    run = 0
    ls = [int(f.split("_")[1]) for f in os.listdir(_dir) if "run_" in f]
    if len(ls) > 0:
        run = max(ls) + 1
    run_path = os.path.join(_dir, "run_{}".format(run))
    os.mkdir(run_path)
    return run_path

def run_dir_from_opt(opt, _dir="experiment_files"):
    ls = [os.path.join(_dir, f) for f in os.listdir(_dir) if "run_" in f]
    for file in ls:
        with open(os.path.join(file, "_info.txt")) as f:
            _info = json.load(f)
        shared_items = len([key for key in opt if key in _info and _info[key] == opt[key]])
        if shared_items < len(opt.keys()):
            continue
        return file, int(file.split("_")[-1])
    print("Didn't find a match")

def load_json(file):
    with open(file) as json_file:
        _dict = json.load(json_file)
    return _dict

def write_to_txt(file, _dict):
    with open(file, "w") as json_file:
        json_file.write(json.dumps(_dict))


def append_json(file, **kwargs):
    old_dict = load_json(file)
    for key, value in kwargs.items():
        old_dict[key] = value
    write_to_txt(file, old_dict)



class StatisticsTracker:
    """
    Tracks the relevant statistics of the trust region optimizer and flushes it to log file.
    """
    def __init__(self,tb_path=None):
        self.statistics = {}
        self.start_time = time.time()
        self.time_offset = 0
        self.writer = SummaryWriter() if tb_path is None else SummaryWriter(log_dir=tb_path)

    def __call__(self, iteration, **kwargs):
        for key, value in kwargs.items():
            if value is None:
                continue

            value = value.item() if isinstance(value, torch.Tensor) else value

            if key not in self.statistics.keys():
                self.statistics[key] = {}
                self.statistics[key]["values"] = [value]
                self.statistics[key]["iteration"] = [iteration]
                self.statistics[key]["time"] = [self.tracked_time()]
            else:
                self.statistics[key]["values"].append(value)
                self.statistics[key]["iteration"].append(iteration)
                self.statistics[key]["time"].append(self.tracked_time())

        # self.save_to_tensorboard()
    def tracked_time(self):
        return time.time() - self.start_time + self.time_offset

    def save_to_tensorboard(self):
        def get_key_prefix(key):
            gradient_norm_group = ["batch_gradient_norm", "full_gradient_norm"]
            loss_group = ["batch_loss", "full_loss"]
            tr_metrics_group = ["fn_improvement", "model_improvement", "sub_steps", "step_norm", "tr_radius"]
            eigenvalue_group = ["minimum_eigenvalue", "minimum_eigenvalue"]
            test_group = ["test_loss", "test_accuracy"]

            if key in gradient_norm_group:
                return "GradientNorm/" + key
            elif key in loss_group:
                return "Loss/" + key
            elif key in tr_metrics_group:
                return "TRMetrics/" + key
            elif key in eigenvalue_group:
                return "Eigenvalue/" + key
            elif key in test_group:
                return "Test/" + key
            else:
                return "Misc/" + key

        # take the last values and store them in a tensorboard file
        for key, statistic in self.statistics.items():
            if statistic is None or len(statistic["values"]) == 0: continue
            store_key = get_key_prefix(key)
            value = statistic["values"][-1]
            n_iter = statistic["iteration"][-1]
            if not isinstance(value, (int, float)): continue
            # print("value: {}, key: {}, n_iter: {}".format(value, store_key, n_iter))
            self.writer.add_scalar(store_key, value, n_iter)

    def to_df(self, y):
        values = {}
        for val_key in y:
            try:
                values[val_key] = self.statistics[val_key]["values"]
            except KeyError:
                continue
        values["iteration"] = self.statistics[y[0]]["iteration"]
        values["time"] = self.statistics[y[0]]["time"]
        df = pd.DataFrame.from_dict(values)
        return df


    def correct_entry(self, **kwargs):
        for key, value in kwargs.items():
            self.statistics[key]["values"][-1] = value

    def reset_time(self):
        self.start_time = time.time()

    def new_time_offset(self):
        self.time_offset += time.time() - self.start_time

    def pause_timer(self):
        self.new_time_offset()

    def restart_timer(self):
        self.reset_time()

    def correct_last_k_statistics(self, k, *args):
        for key in args:
            if key not in self.statistics.keys(): continue
            if len(self.statistics[key]["values"]) <= k: continue
            self.statistics[key]["values"][-k:] = [self.statistics[key]["values"][-k]] * k

    def zero_last_k_statistics(self, k, *args):
        for key in args:
            if key not in self.statistics.keys(): continue
            self.statistics[key]["values"][-k:] = [0] * len(self.statistics[key]["values"][-k:])

    def set_back_to_iteration(self, iteration):
        for key, value in self.statistics.items():
            if not isinstance(value, dict) or "iteration" not in value.keys():
                continue

            if key in ["sample_size", "sub_steps", "tr_radius"]:
                continue

            # find the values that needs to be changed by finding how many iterations are smaller than given iteration
            idx = next((i for i, x in enumerate(value["iteration"]) if x > iteration), -1)
            if idx == -1: continue

            idx_iteration = next((i for i, x in enumerate(value["iteration"]) if x == iteration), -1)
            if idx_iteration == -1: continue

            # set back all the values that came after iteration to the value of iteration
            for i in range(idx, len(value["iteration"])):
                self.statistics[key]["values"][i] = self.statistics[key]["values"][idx_iteration]



class Sampler:
    def __init__(self, train_dset, test_dset=(None, None), seed=42):
        self.X, self.Y = train_dset
        self.X_test, self.Y_test = test_dset
        self.sampled_idxs = RandomIdxList(self.X.size(0), seed=seed)
        self.train_epoch = 0


    def __call__(self, sample_size=None, seed=None, train=True):
        """
        :param sample_size: Can be str, int, float
            - if str then full
            - if int > 1 then sample size is a fixed batch size
            - if float <= 1 then it gives the percentage of datapoints to use
        :param sampling_scheme:
        """
        if seed is not None:
            np.random.seed(seed)

        if isinstance(sample_size, str) and sample_size == "full":
            if train:
                return (self.X, self.Y)
            else:
                return (self.X_test, self.Y_test)
        else:
            N = self.X.size(0) if train else self.X_test.size(0)
            if sample_size <= 1.:
                sample_size = int(sample_size*N)
            else:
                sample_size = int(sample_size)

            sample_size=int(sample_size) 

            if train:
                idx, epoch_count = self.sampled_idxs(sample_size)
                self.train_epoch += epoch_count
                X_ss, Y_ss = torch.index_select(self.X, 0, idx).type(t_FloatTensor), torch.index_select(self.Y, 0, idx).type(t_LongTensor)
            else:
                idx = t_LongTensor(np.random.choice(N, replace=False, size=sample_size))
                X_ss, Y_ss = torch.index_select(self.X_test, 0, idx).type(t_FloatTensor), torch.index_select(self.Y_test, 0, idx).type(t_LongTensor)
            return (X_ss, Y_ss)

    # def reshuffle(self, seed=None):
    #     self.sampled_idxs.reshuffle(seed)


class RandomIdxList:
    def __init__(self, N, seed):
        self.random_state = np.random.RandomState(seed=seed)
        self.idx_list = self.random_state.choice(N, replace=False, size=N)
        self.current_idx = 0
        self.N = N

    def __call__(self, batch_size):
        epoch_count = 0
        if self.current_idx + batch_size >= self.N:
            self.reshuffle()
            epoch_count = 1
        idx_range = range(self.current_idx, self.current_idx+batch_size)
        self.current_idx += batch_size
        return t_LongTensor(self.idx_list[idx_range]), epoch_count

    def reshuffle(self):
        self.idx_list = self.random_state.choice(self.N, replace=False, size=self.N)
        self.current_idx = 0