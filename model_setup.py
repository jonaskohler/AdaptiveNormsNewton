import torch, os
from tensorflow.keras.datasets import mnist, fashion_mnist
from utils import *
from models import FCMNISTModel, MNISTConv, CIFAR10Conv, MNISTAutoencoder, FCCIFARModel
from typing import Optional
import keras
from keras.datasets import cifar10
keras.backend.set_image_data_format('channels_first')

def load_mnist_data(model_name: str, data_path="./data"):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    if model_name == "fc" or model_name =="autoencoder":
        x_train = x_train.reshape([-1, 28 ** 2])
        x_test = x_test.reshape([-1, 28 ** 2])
    elif model_name == "conv":
        x_train = x_train.reshape([-1, 1, 28, 28])
        x_test = x_test.reshape([-1, 1, 28, 28])
    else:
        raise NotImplementedError("Model {} is not supported yet".format(model_name))

    return (x_train, y_train), (x_test, y_test)

def load_fashion_mnist_data(model_name: str, data_path="./data"):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    if model_name == "fc" or model_name =="autoencoder":
        x_train = x_train.reshape([-1, 28 ** 2])
        x_test = x_test.reshape([-1, 28 ** 2])
    elif model_name == "conv":
        x_train = x_train.reshape([-1, 1, 28, 28])
        x_test = x_test.reshape([-1, 1, 28, 28])
    else:
        raise NotImplementedError("Model {} is not supported yet".format(model_name))

    return (x_train, y_train), (x_test, y_test)

def load_cifar_data(model_name: str):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if model_name == "fc" or model_name =="autoencoder":
        x_train = x_train.reshape([-1, 3*(32**2)])
        x_test = x_test.reshape([-1, 3*(32**2)])
    return (x_train, y_train.squeeze()), (x_test, y_test.squeeze())


def load_model(model_name: str, dataset: str):
    if dataset == "mnist" or dataset== "fashion_mnist":
        if model_name == "fc":
            model = FCMNISTModel
        elif model_name == "conv":
            model = MNISTConv
        elif model_name== "autoencoder":
            model = MNISTAutoencoder
        else:
            raise NotImplementedError("Model {} is not supported yet".format(model_name))
    elif dataset == "cifar-10":
        if model_name == "fc":
            model = FCCIFARModel
        elif model_name == "conv":
            model = CIFAR10Conv
        else:
            raise NotImplementedError("Model {} is not supported yet".format(model_name))
    else:
        raise NotImplementedError("Dataset {} is not supported yet".format(dataset))

    return model

def load_loss_fn(loss_name: str):
    def loss(y_pred, target, l2_alpha=0, model=None):
        if loss_name == "CE":
            fn = torch.nn.CrossEntropyLoss()
        elif loss_name == "MSE":
            fn = torch.nn.MSELoss()
        else:
            raise NotImplementedError("Loss function {} is not supported yet".format(loss_name))

        if l2_alpha == 0:
            return fn(y_pred, target)

        # if alpha > 0 then we do l2 regularization on the parameters
        l2_reg = t_FloatTensor(1)
        for w in model.parameters():
            l2_reg = l2_reg + w.norm(2)

        return fn(y_pred, target) + l2_alpha * l2_reg

    return loss


def load_data(dataset_name: str, model_name: str, full_samples: Optional[int] = None):
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = load_mnist_data(model_name)
    elif dataset_name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = load_fashion_mnist_data(model_name)
    elif dataset_name == "cifar-10":
        (x_train, y_train), (x_test, y_test) = load_cifar_data(model_name)
    else:
        raise NotImplementedError("Dataset {} is not supported yet".format(dataset_name))

    full_samples = x_train.shape[0] if full_samples is None else full_samples
    X, Y = torch.from_numpy(x_train[:full_samples]).type(t_FloatTensor), torch.from_numpy(y_train[:full_samples]).type(
        t_LongTensor)

    full_samples_test = min(full_samples, x_test.shape[0])
    X_test, Y_test = torch.from_numpy(x_test[:full_samples_test]).type(t_FloatTensor), torch.from_numpy(
        y_test[:full_samples_test]).type(t_LongTensor)

    train_dset = (X, Y)
    test_dset = (X_test, Y_test)
    return train_dset, test_dset


def setup(opt: dict):
    train_dset, test_dset = load_data(opt["dataset_name"], opt["model_name"], opt["full_samples"])
    model = load_model(opt["model_name"], opt["dataset_name"])
    loss = load_loss_fn(opt["loss_fn"])

    # load the sampler
    sampler = Sampler(train_dset, test_dset, seed=opt["model_seed"])

    print_c("[i] Running {}-model on {} with {} optimizer.".format(opt["model_name"], opt["dataset_name"], opt["method"]), "green")
    # print("[i] Training samples: {} (of dim: {})".format(train_dset[1].size(0), np.prod(train_dset[0].size()[1:])))

    return sampler, model, loss

