from utils.tools import *
from network import *

import os
import paddle
import paddle.optimizer as optim
import time
import numpy as np

# DSH(CVPR2016)
# paper [Deep Supervised Hashing for Fast Image Retrieval](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf)
# code [DSH-pytorch](https://github.com/weixu000/DSH-pytorch)


def get_config():
    config = {
        "lambda": 3,
        # "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5, "betas": (0.9, 0.999)}},
        "optimizer": {"type": optim.RMSProp, "optim_params": {"learning_rate": 1e-5, "weight_decay": 10 ** -5}},

        "info": "[LCDSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": AlexNet,
        # "net":ResNet,

        "dataset": "cifar10-2",
        # "dataset": "nuswide_21",

        "epoch": 150,
        "test_map": 10,

        "device": paddle.set_device("gpu"),
        "bit_list": [48],
        "save_path": "save/LCDSH",
    }
    
    config = config_dataset(config)
    return config

class LCDSHLoss(paddle.nn.Layer):
    """
    # paper [Locality-Constrained Deep Supervised Hashing for Image Retrieval](https://www.ijcai.org/Proceedings/2017/0499.pdf)
    # [LCDSH] epoch:145, bit:48, dataset:cifar10-1,  MAP:0.798, Best MAP: 0.798
    # [LCDSH] epoch:183, bit:48, dataset:nuswide_21, MAP:0.833, Best MAP: 0.834
    """
    def __init__(self, _lambda):
        super(LCDSHLoss, self).__init__()
        self._lambda = _lambda

    def forward(self, feature, label):
        """
        for a batch result:
        feature: features
        label: labels
        index: image index in all the train dataset
        """
        label = label.astype("float32")

        s = 2 * (paddle.matmul(label, label, transpose_y=True) > 0).astype("float32") - 1
        inner_product = paddle.matmul(feature, feature, transpose_y=True) * 0.5

        inner_product = inner_product.clip(min=-50, max=50)
        L1 = paddle.log(1 + paddle.exp(-s * inner_product)).mean()

        #do sgn
        b = feature.sign()
        inner_product_ = paddle.matmul(b, b, transpose_y=True) * 0.5
        sigmoid = paddle.nn.Sigmoid()
        L2 = (sigmoid(inner_product) - sigmoid(inner_product_)).pow(2).mean()

        return L1 + self._lambda * L2

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
    criterion = LCDSHLoss(config["lambda"])

    Best_mAP = 0
    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()
        train_loss = 0
        for image, label, ind in train_loader:
            optimizer.clear_grad()
            u = net(image)   #u is the feature; label

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
                             config["topK"])  #elapsed too log:      4mins/10000 query
            
            #compare with paddle api(map)

            if mAP > Best_mAP:
                Best_mAP = mAP

                if "save_path" in config:
                    if not os.path.exists(config["save_path"]):
                        os.makedirs(config["save_path"])

                    print("save in ", config["save_path"])
                    np.save(os.path.join(config["save_path"], config["dataset"] + str(mAP) + "-" + "trn_binary.npy"),
                            trn_binary.numpy())
                    
                    #save model
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
