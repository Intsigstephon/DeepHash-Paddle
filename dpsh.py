from utils.tools import *
from network import *

import os
import paddle
import paddle.optimizer as optim
import time
import numpy as np

def get_config():
    config = {
        "alpha": 0.1,
        # "optimizer": {"type": optim.SGD, "optim_params": {"lr": 0.005, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSProp, "optim_params": {"learning_rate": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[DPSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": AlexNet,
        # "net":ResNet,
        # "dataset": "cifar10",
        "dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "mirflickr",
        # "dataset": "voc2012",
        # "dataset": "imagenet",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",
        "epoch": 150,
        "test_map": 5,
        "device": paddle.set_device("gpu"),
        "bit_list": [48],
        "save_path": "save/DPSH",
    }
    config = config_dataset(config)
    return config

class DPSHLoss(paddle.nn.Layer):
    """
    paddle reimplementation of DPSHLoss
    paper: Feature Learning based Deep Supervised Hashing with Pairwise Labels(2016)
    """  
    def __init__(self, num_train, n_class, bit, alpha):
        super(DPSHLoss, self).__init__()
        self.U = paddle.zeros([num_train, bit]).astype("float32")
        self.Y = paddle.zeros([num_train, n_class]).astype("float32")
        self.alpha = alpha

    def forward(self, feature, label, index):
        """
        for a batch result:
        feature: features
        label: labels
        index: image index in all the train dataset
        """
        self.U[index, :] = feature
        self.Y[index, :] = label

        s = (paddle.matmul(label, self.Y, transpose_y=True) > 0).astype("float32")
        inner_product = paddle.matmul(feature, self.U, transpose_y=True) * 0.5

        #likelihood loss
        likelihood_loss = ((1 + (-inner_product.abs()).exp()).log() + inner_product.clip(min=0) - s * inner_product).mean()
        
        #quantization_loss
        quantization_loss = (feature - feature.sign()).pow(2).mean()

        return likelihood_loss + quantization_loss * self.alpha

def train_val(config, bit):
    device = config["device"]

    #diff in dataloader; difference appear here
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)  

    config["num_train"] = num_train
    net = config["net"](bit, "./pretrain/AlexNet_pretrained")

    # import pdb
    # pdb.set_trace()
    multi_label = "nuswide" in config["dataset"]
    optimizer = config["optimizer"]["type"](parameters = net.parameters(), rho=0.99, epsilon=1e-08, **(config["optimizer"]["optim_params"]))
    criterion = DSHLoss(config["num_train"], config["n_class"], bit, config["alpha"])

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



