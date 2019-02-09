import torch
import torch.nn as nn
import numpy as np


class CINBrock(nn.Module):

    def __init__(self, h_next, h_prev, field_num):
        super(CINBrock, self).__init__()
        self.conv = nn.Conv1d(1, h_next, kernel_size=h_prev *
                              field_num, stride=h_prev*field_num)

    def forward(self, x0, x):
        batchsize = x0.size(0)
        # x0 shape:(batchsize, m, d)
        # x_prev shape:(batchsize, Hk-1, d)
        x = x0[:, :, None] * x[:, None, :]
        # x shape:(B, Hk-1, m, D)
        x = x.view(batchsize, 1, -1)
        x = self.conv(x)
        # x shape:(B, Hk, D), Hk is out_channels of self.conv

        return x


class CIN(nn.Module):

    def __init__(self, input_size, out_size, embed_dim, field_num, hks, brock_num=3):
        assert len(hks) == brock_num, print(
            "length of 'k_params'({}) must be equal to 'brock_num'({})").format(len(hks), brock_num)
        super(CIN, self).__init__()

        self.cin_brocks = []
        self.h_sum = 0
        prev_hk = field_num
        for cin_brock_id, cin_brock in enumerate(range(brock_num)):
            hk = hks[cin_brock_id]
            self.h_sum += hk
            self.cin_brocks.append(CINBrock(
                hk, prev_hk, field_num).to())
            self.add_module("cin_brock_{}".format(cin_brock_id),
                            self.cin_brocks[cin_brock_id])
            prev_hk = hk

        self.linear = nn.Linear(self.h_sum, out_size)

    def forward(self, x0):
        batchsize = x0.size(0)
        x_h = x0.clone()
        ps = []
        for cin_brock in self.cin_brocks:
            x_h = cin_brock(x0, x_h)
            # x_h shape:(B, Hk, D)
            p = torch.sum(x_h, dim=2)
            ps.append(p)

            # p shape:(B, Hk)
        ps = torch.cat(ps, dim=1)
        # shape: (B, h_sum)

        return self.linear(ps)


class DNN(nn.Module):

    def __init__(self, out_size,  embed_dim, field_num, n_hiddens, layer_num=3, activation=nn.ReLU()):
        super(DNN, self).__init__()
        assert len(n_hiddens) == layer_num, print(
            "length of 'k_params'({}) must be equal to 'brock_num'({})").format(len(n_hiddens), layer_num)
        self.activation = activation
        self.linears = []
        prev_hidden = field_num * embed_dim
        for layer in range(layer_num-1):
            self.linears.append(
                nn.Linear(prev_hidden, n_hiddens[layer]))
            prev_hidden = n_hiddens[layer]
            self.add_module("linear_{}".format(layer), self.linears[layer])
        self.last_linear = nn.Linear(n_hiddens[-1], out_size)

    def forward(self, x):
        # x shape:(B, m*D)
        for linear in self.linears[:-1]:
            # print("x size in dnn:", x.size())
            x = linear(x)
            x = self.activation(x)
        x = self.last_linear(x)

        return x


# class Linear(nn.Module):

#     def __init__(self, input_size, out_size):
#         super(Linear, self).__init__()
#         self.linear = nn.Linear(input_size, out_size)

#     def forward(self, x):

#         return self.linear(x)


class xDeepFM(nn.Module):

    def __init__(self, input_size, out_size, field_num, embed_dim, dnn_params, cin_params, activation, device="cpu"):
        dnn_layer_num = dnn_params["layer_num"]
        dnn_n_hiddens = dnn_params["n_hidden"]
        cin_layer_num = cin_params["layer_num"]
        cin_hks = cin_params["hk"]
        activation_dict = {
            "relu": nn.ReLU(),
            "lrelu": nn.LeakyReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh()
        }
        super(xDeepFM, self).__init__()
        self.embed = nn.Embedding(input_size, embed_dim)
        # use for onehot vector * W, to save memory
        self.linear_w_by_embed = nn.Embedding(input_size, out_size)
        self.cin = CIN(input_size, out_size, embed_dim,
                       field_num, cin_hks, brock_num=cin_layer_num)
        self.dnn = DNN(out_size, embed_dim, field_num,
                       dnn_n_hiddens, layer_num=dnn_layer_num, activation=activation_dict[activation])
        # for k in self.children():
        #     print(k, "DFAd")
        # self.linear = Linear(input_size, out_size).to(device)

    def forward(self, x):
        batchsize = x.size(0)
        x_linear = self.linear_w_by_embed(x)
        # x_linear shape:(B,field_num, out_size)
        x_linear = x_linear.sum(dim=1)
        x0 = self.embed(x)
        x_cin = self.cin(x0)
        x_dnn = self.dnn(x0.view(batchsize, -1))
        # print("linear size:", x_linear.size())
        # print("cin size:", x_cin.size())
        # print("dnn size:", x_dnn.size())

        return x_linear + x_cin + x_dnn


if __name__ == "__main__":
    pass
