import numpy as np
from PIL import Image
from tqdm import tqdm
import paddle
from paddle.vision.datasets import Cifar10
from paddle.vision import transforms

def config_dataset(config):
    """
    the configure of dataset:
    1. n_classes
    2. topK
    3. datapath
    4. dataset: Actually, it is the name
    support many dataset: 
    cifar, nuswide_21, nuswide_21_m, nuswide_81_m, coco, imagenet, mirflicker, voc2012 
    """
    if "cifar" in config["dataset"]:
        config["topK"] = -1
        config["n_class"] = 10

    elif config["dataset"] in ["nuswide_21", "nuswide_21_m"]:
        config["topK"] = 5000
        config["n_class"] = 21

    elif config["dataset"] == "nuswide_81_m":
        config["topK"] = 5000
        config["n_class"] = 81

    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80

    elif config["dataset"] == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100

    elif config["dataset"] == "mirflickr":
        config["topK"] = -1
        config["n_class"] = 38

    elif config["dataset"] == "voc2012":
        config["topK"] = -1
        config["n_class"] = 20

    config["data_path"] = "/dataset/" + config["dataset"] + "/"   #by default
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "/dataset/NUS-WIDE/"

    if config["dataset"] in ["nuswide_21_m", "nuswide_81_m"]:
        config["data_path"] = "/dataset/nus_wide_m/"

    if config["dataset"] == "coco":
        config["data_path"] = "/dataset/COCO_2014/"

    if config["dataset"] == "voc2012":
        config["data_path"] = "/dataset/"

    config["data"] = {
        "train_set": {"list_path": "./data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}

    return config

class ImageList(object):
    def __init__(self, data_path, image_list, transform):
        #support multi label
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

      
def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]

    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])

class MyCIFAR10(Cifar10):
    def __getitem__(self, index):
        img, target = self.data[index] 
        img = np.reshape(img, [3, 32, 32])   
        img = img.transpose([1, 2, 0]).astype("uint8") 

        img = Image.fromarray(img)
        img = self.transform(img)  #3 * 224 * 224;  double, (-1, 1)

        target = np.eye(10, dtype=np.int8)[np.array(target)]   #to one-hot
        return img, target, index

def get_index(dataset, label):
    rslt = []
    for i in range(len(dataset)):
        if dataset[i][1] == label:
            rslt.append(i)
    return np.array(rslt)
  
def cifar_dataset(config):
    """
    total: 60000:  10 * 6000
    train: 50000 = 10 * 5000
    test:  10000 = 10 * 1000
    """
    batch_size = config["batch_size"]
    
    train_size = 500
    test_size = 100

    if config["dataset"] == "cifar10-2":
        train_size = 5000
        test_size  = 1000

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = MyCIFAR10(data_file=None,
                              mode="train",
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(data_file=None,
                             mode="test",
                             transform=transform)

    database_dataset = MyCIFAR10(data_file=None,
                                 mode="test",
                                 transform=transform)

    X = train_dataset.data
    X.extend(test_dataset.data)

    first = True
    for label in range(10):
        index = get_index(X, label)  
        N = index.shape[0]
        perm = np.random.permutation(N)  

        index = index[perm]

        if first:
            test_index  = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config["dataset"] == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif config["dataset"] == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif config["dataset"] == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index

    train_dataset.data = np.array(X)[train_index]
    test_dataset.data =  np.array(X)[test_index]
    database_dataset.data = np.array(X)[database_index]

    print("train_dataset",    train_dataset.data.shape[0])
    print("test_dataset",     test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    train_dataset.data = list(train_dataset.data)
    test_dataset.data  = list(test_dataset.data)
    database_dataset.data = list(database_dataset.data)

    train_loader = paddle.io.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = paddle.io.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    database_loader = paddle.io.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4)


    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]

def get_data(config):
    """
    support cifar-0/1/2 and other dataset
    return train_loader, test_loader, database_loader
    len of train/test/database
    """
    if "cifar" in config["dataset"]:
        return cifar_dataset(config)

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = paddle.io.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle=True, num_workers=4)

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])

def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()

    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img)))
    return paddle.concatenate(bs).sign(), paddle.concatenate(clses)

draw_range = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
              9000, 9500, 10000]

def pr_curve(rF, qF, rL, qL, draw_range=draw_range):
    #  https://blog.csdn.net/HackerTom/article/details/89425729
    n_query = qF.shape[0]
    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    Rank = np.argsort(CalcHammingDist(qF, rF))
    P, R = [], []
    for k in tqdm(draw_range):
        p = np.zeros(n_query)
        r = np.zeros(n_query)
        for it in range(n_query):
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]
            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)
            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all
        P.append(np.mean(p))
        R.append(np.mean(r))
    return P, R

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

if __name__=="__main__":

    import paddle.optimizer as optim

    config = {
        "alpha": 0.1,
        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSProp, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[DSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        #"net": AlexNet,
        #"net": ResNet,

        "dataset": "cifar10",
        # "dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "mirflickr",
        # "dataset": "voc2012",
        # "dataset": "imagenet",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",

        "epoch": 250,
        "test_map": 15,
        "save_path": "save/DSH_resnet",
        "bit_list": [48],
    }

    get_data(config)

