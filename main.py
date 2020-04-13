from utils import *
from trust_region import Ellipsoidal_TR
from model_setup import setup
from sgd import StandardOptimization
from plotter import Plotter


class ParameterLoader():
    def __init__(self, env_path, method):
        """
        :param env_path: Path to the standard env parameters.
        :param method_path: If str: Path to the standard method parameters.
                            If dict: dictionary of standard method parameters
        """
        self.standard_env = load_json(env_path)

        if isinstance(method, str):
            self.standard_method = load_json(method)
        else:
            self.standard_method = method

        # merge the two dictionaries
        self.opt = {**self.standard_env, **self.standard_method}

    def overwrite_params(self, opt):
        for key, value in opt.items():
            self.opt[key] = value

    def __call__(self):
        return self.opt

class Trainer:
    def __init__(self, env_path:dict, method_path:dict, changes:dict=None):
        # Load the parameter files
        param_loader = ParameterLoader(env_path, method_path)
        if changes is not None:
            param_loader.overwrite_params(changes)
        self.opt = param_loader()

        self.setup()


    def run(self, y=["full_loss", "sub_steps"]):

        if not use_gpu():
            print_c("[i] No CUDA-capable device found. Running the algorithms on CPU significantly slows down training.", "red")
            time.sleep(1)

        if self.opt["method"] == "TR":
            tr = Ellipsoidal_TR(self.model, self.sampler, self.loss, opt=self.opt)
            stats = tr(max_iterations=self.opt["max_iterations"])

        elif (self.opt["method"] == "SGD" or self.opt["method"] == "Adagrad" or self.opt["method"] == "Rmsprop"):
            sgd = StandardOptimization(self.model, self.sampler, self.loss, self.opt)
            stats = sgd(max_iterations=self.opt["max_iterations"])

        else:
            raise NotImplementedError

        return stats.to_df(y)

    def setup(self):
        method = self.opt["method"]
        if method == "TR":
            method += "_{}".format(self.opt["preconditioner"])
        setup_options = {
            "dataset_name": self.opt["dataset"],
            "full_samples": self.opt["full_samples"],
            "model_name": self.opt["model_name"],
            "loss_fn": self.opt["loss_fn"],
            "model_seed": self.opt["model_seed"],
            "method": method
        }

        self.sampler, model, self.loss = setup(setup_options)
        if use_gpu():
            self.model = model(seed=self.opt["model_seed"]).cuda()
        else:
            self.model = model(seed=self.opt["model_seed"])
        time.sleep(1)

        # print(num_parameters(self.model))

class Environment:
    def __init__(self, method_names:list, model_name:str, dataset:str, size="standard"):
        self.model_name = self.map_model_name(model_name)
        self.method_names = self.map_method_names(method_names)
        self.dataset = dataset

        self.env = "parameters/environment/{}/{}_{}.txt".format(self.dataset, self.model_name, size)
        method_base = "parameters/method/mnist/fc"
        self.method_paths = {method: os.path.join(method_base, "{}.txt".format(method)) for method in self.method_names}
        self.df = None


    def map_method_names(self, method_names):
        possible_methods = ["TR_uniform", "TR_adagrad", "TR_rms", "Rmsprop", "Adagrad", "SGD"]
        for method in method_names:
            if method not in possible_methods:
                raise NotImplementedError("No method with name: {}. Please use one of the following {}.".format(method, possible_methods))
        return method_names


    def map_model_name(self, model_name):
        if model_name == "conv":
            return "conv"
        elif model_name == "mlp":
            return "fc"
        else:
            raise NotImplementedError("No model with name: {}".format(model_name))


    def run_training(self, changes:None):
        if changes is None:
            changes = {method:{} for method in self.method_names}
        for method in self.method_names:
            if method in changes.keys(): continue
            changes[method] = {}

        df = None
        for method_name, method_path in self.method_paths.items():
            df_new = Trainer(self.env, method_path, changes=changes[method_name]).run()
            df_new["method"] = method_name

            if df is None:
                df = df_new
            else:
                df = df.append(df_new, ignore_index=True, sort=False)
        self.df = self.adjust_df(df)

    def plot(self, x="time"):
        Plotter.plot(self.df, x)

    def adjust_df(self, df):
        def compute_backprops(df):
            df = df.sort_values(["method", "iteration"])
            df["backprops"] = 1.

            def reset_lasts():
                last_iter = 0
                last_sub_steps = 1.
                last_bp = 0
                return last_iter, last_sub_steps, last_bp

            last_iter, last_sub_steps, last_bp = reset_lasts()
            for index, row in df.iterrows():
                if "TR" in row.method:
                    if row.iteration == 0:
                        df.loc[index, "backprops"] = 0
                        df.loc[index, "sub_steps"] = max(df.loc[index, "sub_steps"], 1.)
                        last_iter, last_sub_steps, last_bp = reset_lasts()
                    else:
                        assert (df.loc[index, "iteration"] - last_iter) > 0
                        df.loc[index, "backprops"] = (df.loc[index, "iteration"] - last_iter) * last_sub_steps + last_bp
                        df.loc[index, "sub_steps"] = max(df.loc[index, "sub_steps"], 1.)
                        last_iter = df.loc[index, "iteration"]
                        last_sub_steps = df.loc[index, "sub_steps"]
                        last_bp = df.loc[index, "backprops"]
                else:
                    df.loc[index, "backprops"] = df.loc[index, "iteration"]
            return df

        df["log_loss"] = [np.log(val) for val in df.full_loss]

        df = compute_backprops(df)
        return df


