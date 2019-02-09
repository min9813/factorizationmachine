import torch
import torch.nn as nn
import argparse
import os
import xdeepfm
import matplotlib.pyplot as plt
from torch import optim
from mldataset import MLDataset
from torch.utils.data import DataLoader
from collections import namedtuple

MLPATH = "/home/minteiko/developer/project/data/ml-20m"
SMALL_DATAPATH = os.path.join(MLPATH, "small_ratings.csv")


def make_dataset(data_path, batchsize=128, valid_size=30000):
    dataset = MLDataset(data_path)
    meta_config = {
        "feature_num": dataset.feature_num,
        "field_num": dataset.field_num
    }
    valid_data = dataset[-valid_size:]
    train_data = dataset[:-valid_size]
    valid_iter = DataLoader(valid_data, batch_size=batchsize, shuffle=False)
    train_iter = DataLoader(train_data, batch_size=batchsize, shuffle=True)

    return train_iter, valid_iter, meta_config


def load_config_file(config_path):
    # print(config_path)
    with open(config_path, "r") as f:
        configuration = f.readlines()
    dnn_params = {}
    cin_params = {}
    params = {}
    for conf in configuration:
        if conf.startswith("#"):
            continue
        conf_key, conf_value = conf.strip().split(":")
        try:
            if "," in conf_value:
                conf_value = list(map(int, conf_value.split(",")))
            else:
                conf_value = int(conf_value)
        except ValueError:
            pass
        if "dnn" in conf_key:
            if "layer_num" in conf_key:
                dnn_params["layer_num"] = conf_value
            else:
                dnn_params[conf_key[4:]] = conf_value
        elif "cin" in conf_key:
            if "layer_num" in conf_key:
                cin_params["layer_num"] = conf_value
            else:
                cin_params[conf_key[4:]] = conf_value
        else:
            params[conf_key] = conf_value
    params["cin_params"] = cin_params
    params["dnn_params"] = dnn_params

    return params


def make_model(params, field_num, input_size, output_size=1, device="cpu"):
    model = xdeepfm.xDeepFM(
        input_size, output_size, field_num, params["embed_dim"], params["dnn_params"], params["cin_params"], params["activation"], device=device)

    return model


def setup(config_path, data_path, batchsize, valid_size, weight_decay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = load_config_file(config_path)
    train_iter, valid_iter, data_meta_config = make_dataset(
        data_path, batchsize=batchsize, valid_size=valid_size)
    model = make_model(
        params,  data_meta_config["field_num"], data_meta_config["feature_num"], output_size=1, device=device).to(device)
    mseloss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3,
                           weight_decay=weight_decay)

    train_models = namedtuple(
        "TrainModel", ("model", "loss", "optimizer", "train_iter", "valid_iter"))
    return train_models(model, mseloss, optimizer, train_iter, valid_iter)


def train(model, criterion, train_data, valid_data, optimizer, max_iter, log_interval=100, test_data=None):
    model.train()
    valid_loss = []
    train_loss = []
    for epoch in range(max_iter):
        for batch_idx, x_batch in enumerate(train_data):
            x_batch, y_batch = x_batch[:, :-1], x_batch[:, -1].cuda()
            x_batch = x_batch.type(torch.LongTensor).cuda()
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t MSE Loss: {:.6f}'.format(
                    epoch, batch_idx * len(x_batch), len(train_data.dataset),
                    100. * batch_idx / len(train_data), loss.item()))
        train_loss.append(loss.item())
        valid_loss.append(evaluator(model, criterion, valid_data))

    return train_loss, valid_loss


def evaluator(model, criterion, valid_data):
    model.eval()
    with torch.no_grad():
        loss = 0
        for batch_idx, x_batch in enumerate(valid_data):
            x_batch, y_batch = x_batch[:, :-1], x_batch[:, -1].cuda()
            x_batch = x_batch.type(torch.LongTensor).cuda()
            output = model(x_batch)
            # print(batch_idx)
            # print(output.size(), y_batch.size())
            cur_loss = criterion(output.squeeze(), y_batch)
            loss += cur_loss
        loss = loss.item()/batch_idx
        print('Valid MSE Loss: {:.6f}'.format(
            loss,))
    model.train()
    return loss


def plot_result(train_loss, valid_loss, save_fig_folder):
    length = len(train_loss)
    if os.path.exists(save_fig_folder) is False:
        os.mkdir(save_fig_folder)
    plt.plot(range(length), train_loss, label="t_loss")
    plt.plot(range(length), valid_loss, label="v_loss")
    plt.legend()
    plt.title("loss")
    plt.savefig(save_fig_folder+"/loss.png")


def main():
    if args.conf_path.startswith("./model.conf"):
        conf_path = os.path.abspath(
            os.path.join(__file__, "../", args.conf_path))
    else:
        conf_path = args.conf_path
    train_models = setup(conf_path, args.data_path,
                         args.batchsize, args.valid_size, args.weight_decay)
    train_loss, valid_loss = train(train_models.model, train_models.loss, train_models.train_iter,
                                   train_models.valid_iter, train_models.optimizer, args.max_epoch)
    plot_result(train_loss, valid_loss, args.save_fig_folder)


parser = argparse.ArgumentParser(
    description="This file is used to train semi-supervised model")
parser.add_argument("-b", "--batchsize",
                    default=1024, type=int)
parser.add_argument("--max_epoch",
                    help="max epoch",
                    default=100, type=int)
parser.add_argument("-d", "--data_path",
                    help="path to data", default=SMALL_DATAPATH)
parser.add_argument("-c", "--conf_path",
                    help="path to config file", default="./model.conf")
parser.add_argument("-v", "--valid_size",
                    help="size of validation data", default=30000, type=int)
parser.add_argument("-wd", "--weight_decay",
                    help="term of parametor regularization", default=1e-4,
                    type=int)
parser.add_argument("-li", "--log_interval",
                    help="interval of iteration to display log",
                    default=100,
                    type=int)
parser.add_argument("-f", "--save_fig_folder",
                    help="path to save loss figure",
                    default=os.path.abspath(
                        os.path.join(__file__, "../result")),
                    )

args = parser.parse_args()

if __name__ == "__main__":
    # print(os.getcwd(), __file__)
    main()
