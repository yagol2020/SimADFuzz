import json
import os
import socket

import torch.nn as nn
import numpy as np


def sdc_feature_extract(temp_list, json_dir) -> json:
    SDC_MODEL_FEATURES = 18
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("localhost", 12555))
    path = f"{json_dir}.sdc_coord.json"
    path = os.path.abspath(path)
    if isinstance(temp_list, np.ndarray):
        temp_list = temp_list.tolist()
    json.dump(temp_list, open(path, "w"))
    path = path.encode()
    client_socket.sendall(path)
    data = client_socket.recv(9999999)
    recv_data = json.loads(data.decode())
    client_socket.close()
    if "error" in recv_data:
        return [0] * SDC_MODEL_FEATURES
    recv_data.pop("test_duration")
    recv_data.pop("safety")
    f = []
    for k, v in recv_data.items():
        f.append(v[0])
    assert len(f) == SDC_MODEL_FEATURES
    return f


# 构建Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, car_num, car_features_size, dropout_prob=0.2, hidden_dim=128, n_layers=2, output_dim=1):
        super(TransformerModel, self).__init__()

        self.input_dim = car_num * car_features_size

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=car_num,
                                                        dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout_prob)

        self.fc_out = nn.Linear(self.input_dim, output_dim)

    def forward(self, x, return_features=False):
        x = x.permute(1, 0, 2)
        output = self.transformer_encoder(x)
        output = output[-1, :, :]
        output = self.dropout(output)
        if return_features:
            return output
        output = self.fc_out(output)
        return output
