from utils.tools import *
from network import *

import os
import paddle
import paddle.optimizer as optim
import time
import numpy as np

# DSHSD(IEEE ACCESS 2019)
# paper [Deep Supervised Hashing Based on Stable Distribution](https://ieeexplore.ieee.org/document/8648432/)
# [DSHSD] epoch:70, bit:48, dataset:cifar10-1, MAP:0.809, Best MAP: 0.809
# [DSHSD] epoch:250, bit:48, dataset:nuswide_21, MAP:0.809, Best MAP: 0.815
# [DSHSD] epoch:135, bit:48, dataset:imagenet, MAP:0.647, Best MAP: 0.647

"""
实验结果：
1. cifar:   5000、1000、54000      0.766
2. cifar-1: 5000、1000、59000      0.806(45 epoch)
3. cifar-2: 50000、10000、50000    0.820
4. nus-wide-21:  train_set 10500;  test 2100; database 193734     0.940
"""

def get_config():
    config = {
        "alpha": 0.05,
        "optimizer": {"type": optim.Adam, "optim_params": {"learning_rate": 1e-5, "beta1": 0.9, "beta2":0.999}},
        "info": "[DSHSDSD]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": AlexNet,
        # "net":ResNet,

        "dataset": "cifar10",
        # "dataset": "imagenet",
        # "dataset": "nuswide_21",
        "epoch": 250,
        "test_map": 15,
        "device": paddle.set_device("gpu"),
        "bit_list": [48],
        "save_path": "save/DSHSD"
    }
    config = config_dataset(config)
    return config

class DSHSDLoss(paddle.nn.Layer):
    """
    # DSHSD(IEEE ACCESS 2019)
    # paper [Deep Supervised Hashing Based on Stable Distribution](https://ieeexplore.ieee.org/document/8648432/)
    # [DSHSD] epoch:70,  bit:48,  dataset:cifar10-1, MAP:0.809, Best MAP: 0.809
    # [DSHSD] epoch:250, bit:48, dataset:nuswide_21, MAP:0.809, Best MAP: 0.815
    # [DSHSD] epoch:135, bit:48, dataset:imagenet, MAP:0.647, Best MAP: 0.647
    """
    def __init__(self, n_class, bit, alpha, multi_label=False):
        super(DSHSDLoss, self).__init__()
        self.m = 2 * bit     
        
        self.fc = paddle.nn.Linear(bit, n_class, bias_attr=False)
        self.alpha = alpha
        self.multi_label = multi_label
        self.n_class = n_class

    def forward(self, feature, label):
        """
        for a batch result:
        feature: features
        label: labels
        """
        feature = feature.tanh().astype("float32")

        dist = paddle.sum(
                    paddle.square((paddle.unsqueeze(feature, 1) - paddle.unsqueeze(feature, 0))), 
                    axis=2)
        
        #label to onehot
        # label = paddle.flatten(label)
        # label = paddle.nn.functional.one_hot(label,  self.n_class).astype("float32")

        s = (paddle.matmul(label, label, transpose_y=True) == 0).astype("float32")
        Ld = (1 - s) / 2 * dist + s / 2 * (self.m - dist).clip(min=0)
        Ld = Ld.mean()
        
        logits = self.fc(feature)
        if self.multi_label:
            # formula 8, multiple labels classification loss
            Lc = (logits - label * logits + ((1 + (-logits).exp()).log())).sum(axis=1).mean()
        else:
            # formula 7, single labels classification loss
            Lc = (-paddle.nn.functional.softmax(logits).log() * label).sum(axis=1).mean()

        return Lc + Ld * self.alpha

def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)

    config["num_train"] = num_train
    net = config["net"](bit, "./pretrain/AlexNet_pretrained")

    optimizer = config["optimizer"]["type"](parameters = net.parameters(), **(config["optimizer"]["optim_params"]))
    criterion = DSHSDLoss(config["n_class"], bit, config["alpha"])

    Best_mAP = 0
    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()
        train_loss = 0
        for image, label, ind in train_loader:

            optimizer.clear_grad()
            u = net(image) 

            loss = criterion(u, label)
            train_loss += loss.numpy()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        if (epoch + 1) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net, device=device)
            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = compute_result(dataset_loader, net, device=device)
            # print("calculating map.......")
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])

            if mAP > Best_mAP:
                Best_mAP = mAP

                if "save_path" in config:
                    if not os.path.exists(config["save_path"]):
                        os.makedirs(config["save_path"])

                    print("save in ", config["save_path"])
                    np.save(os.path.join(config["save_path"], config["dataset"] + str(mAP) + "-" + "trn_binary.npy"),
                            trn_binary.numpy())
                    
                    paddle.save(net.state_dict(),
                               os.path.join(config["save_path"], config["dataset"] + "-" + str(mAP) + "-model.pdparams"))

            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], mAP, Best_mAP))
            print(config)

if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)
