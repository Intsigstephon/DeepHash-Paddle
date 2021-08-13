import paddle
import paddle.nn as nn

#2016
class DSHLoss2(paddle.nn.Layer):
    """
    paddle reimplementation of DSH Loss
    Paper: Deep Supervised Hashing for Fast Image Retrieval(2016)
    """
    def __init__(self, num_train, n_class, bit, alpha):
        super(DSHLoss2, self).__init__()
        self.m = 2 * bit
        self.U = paddle.zeros([num_train, bit]).astype("float64")
        self.Y = paddle.zeros([num_train, n_class]).astype("float64")
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
        dist = paddle.sum(
                    paddle.square((paddle.unsqueeze(feature, 1) - paddle.unsqueeze(self.U, 0))), 
                    axis=2)
        
        y = (paddle.matmul(label, self.Y, transpose_y=True) == 0).astype("float32")
        loss = ((1 - y) / 2 * dist + y / 2 * (self.m - dist).clip(min=0)).mean()
        loss_reg = (1 - feature.sign()).abs().mean()

        return loss + loss_reg * self.alpha

class DHNLoss2(paddle.nn.Layer):
    """
    paddle reimplementation of DHNLoss
    paper: Deep Hashing Network for Efficient Similarity Retrieval(2016)
    """
    def __init__(self, num_train, n_class, bit, alpha):
        super(DHNLoss2, self).__init__()
        self.U = paddle.zeros([num_train, bit]).astype("float64")
        self.Y = paddle.zeros([num_train, n_class]).astype("float64")
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

        s = (paddle.matmul(label, self.Y, transpose_y=True) > 0).astype("float64")
        inner_product = paddle.matmul(feature, self.U, transpose_y=True) * 0.5

        #likelihood loss
        likelihood_loss = ((1 + (-inner_product.abs()).exp()).log() + inner_product.clip(min=0) - s * inner_product).mean()
        
        #quantization_loss
        quantization_loss = (feature.abs() - 1).cosh().log().mean()
        #quantization_loss = (u.abs() - 1).abs().mean()

        return likelihood_loss + quantization_loss * self.alpha

class DPSHLoss2(paddle.nn.Layer):
    """
    paddle reimplementation of DPSHLoss
    paper: Feature Learning based Deep Supervised Hashing with Pairwise Labels(2016)
    """  
    def __init__(self, num_train, n_class, bit, alpha):
        super(DPSHLoss2, self).__init__()
        self.U = paddle.zeros([num_train, bit]).astype("float64")
        self.Y = paddle.zeros([num_train, n_class]).astype("float64")
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

        s = (paddle.matmul(label, self.Y, transpose_y=True) > 0).astype("float64")
        inner_product = paddle.matmul(feature, self.U, transpose_y=True) * 0.5

        #likelihood loss
        likelihood_loss = ((1 + (-inner_product.abs()).exp()).log() + inner_product.clip(min=0) - s * inner_product).mean()
        
        #quantization_loss
        quantization_loss = (feature - feature.sign()).pow(2).mean()

        return likelihood_loss + quantization_loss * self.alpha

class DTSHLoss2(paddle.nn.Layer):
    """
    paddle reimplementation of DPSHLoss
    paper [Deep Supervised Hashing with Triplet Labels](https://arxiv.org/abs/1612.03900)
    code  [DTSH](https://github.com/Minione/DTSH)
    """
    def __init__(self, alpha, _lambda):
        super(DTSHLoss2, self).__init__()
        self.alpha = alpha
        self._lambda = _lambda

    def forward(self, feature, label, index):
        """
        for a batch result:
        feature: features
        label: labels
        index: image index in all the train dataset
        """
        s = (paddle.matmul(label, label, transpose_y=True) > 0).astype("int32")
        inner_product = paddle.matmul(feature, feature, transpose_y=True)
        
        #triplet_loss
        pair_count = 0
        triplet_loss = 0
        for row in range(s.shape[0]):
            if s[row].sum() != 0 and (1 - s[row]).sum() != 0:
                pair_count += 1
                theta_positive = paddle.masked_select(inner_product[row], s[row] == 1)
                theta_negative = paddle.masked_select(inner_product[row], s[row] == 0)
                triple = (theta_positive.unsqueeze(1) - theta_negative.unsqueeze(0) - self.alpha).clip(min=-100, max=50)
                triplet_loss += - (triple - (1 + triple.exp()).log()).mean()

        if pair_count != 0:
            triplet_loss = triplet_loss / pair_count

        #quantization_loss
        quantization_loss = (feature - feature.sign()).pow(2).mean()

        return triplet_loss + quantization_loss * self._lambda

class HashNetLoss2(paddle.nn.Layer):
    """
    paddle reimplementation of HashNetLoss
    # paper [HashNet: Deep Learning to Hash by Continuation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf)
    # code [HashNet caffe and pytorch](https://github.com/thuml/HashNet)
    """
    def __init__(self, num_train, n_class, bit, alpha):
        super(HashNetLoss2, self).__init__()
        self.U = paddle.zeros([num_train, bit]).astype("float64")
        self.Y = paddle.zeros([num_train, n_class]).astype("float64")
        self.alpha = alpha

    def forward(self, feature, label, index, scale=1):
        """
        for a batch result:
        feature: features
        label: labels
        index: image index in all the train dataset
        """        
        feature = (scale * feature).tanh()
        self.U[index, :] = feature
        self.Y[index, :] = label
        
        #get similarity matrix
        similarity = (paddle.matmul(label, self.Y, transpose_y=True) > 0).astype("float64")

        #get product matrix
        dot_product = paddle.matmul(feature, self.U, transpose_y=True) * self.alpha

        mask_positive = (similarity > 0).astype("float64")
        mask_negative = (similarity <= 0).astype("float64")

        #loss
        exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clip(min=0) - similarity * dot_product

        #weight
        S1 = mask_positive.sum()
        S0 = mask_negative.sum()
        S = S0 + S1
        
        #update
        mask = (mask_positive * S / S1) + (mask_negative * S / S0)
        exp_loss = exp_loss * mask
        loss = exp_loss.sum() / S
        return loss

##todo: updataW is not reimplemented
class DSDHLoss2(paddle.nn.Layer):
    """
    # paper [Deep Supervised Discrete Hashing](https://papers.nips.cc/paper/6842-deep-supervised-discrete-hashing.pdf)
    # code  [DSDH_PyTorch](https://github.com/TreezzZ/DSDH_PyTorch)
    """
    def __init__(self, num_train, n_class, bit, mu, nu):
        super(DSDHLoss2, self).__init__()
        self.U = paddle.zeros([bit, num_train]).astype("float64")
        
        #as int(0 or 1)
        self.B = paddle.zeros([bit, num_train]).astype("float64")
        self.Y = paddle.zeros([n_class, num_train]).astype("float64")
        self.mu = mu
        self.nu = nu

    def forward(self, feature, label, index):
        """
        for a batch result:
        feature: features
        label: labels
        index: image index in all the train dataset
        """
        self.U[:, index] = feature.t()
        self.Y[:, index] = label.t()

        #get inner product
        inner_product = paddle.matmul(feature, self.U) * 0.5
        s = (paddle.matmul(label, self.Y) > 0).astype("float64")

        #likelihood
        likelihood_loss = ((1 + (-inner_product.abs()).exp()).log() + inner_product.clip(min=0) - s * inner_product).mean()

        #classification loss
        cl_loss = (y.t() - paddle.matmul(self.W.t(), self.B[:, ind])).pow(2).mean()

        #regularization loss
        reg_loss = self.W.pow(2).mean()

        loss = likelihood_loss + self.mu * cl_loss + self.nu * reg_loss
        return loss

class LCDSHLoss2(paddle.nn.Layer):
    """
    # paper [Locality-Constrained Deep Supervised Hashing for Image Retrieval](https://www.ijcai.org/Proceedings/2017/0499.pdf)
    # [LCDSH] epoch:145, bit:48, dataset:cifar10-1,  MAP:0.798, Best MAP: 0.798
    # [LCDSH] epoch:183, bit:48, dataset:nuswide_21, MAP:0.833, Best MAP: 0.834
    """
    def __init__(self, _lambda):
        super(LCDSHLoss2, self).__init__()
        self._lambda = _lambda

    def forward(self, feature, label):
        """
        for a batch result:
        feature: features
        label: labels
        index: image index in all the train dataset
        """
        s = 2 * paddle.matmul(label, label, transpose_y=True) - 1
        inner_product = paddle.matmul(feature, feature, transpose_y=True) * 0.5

        inner_product = inner_product.clip(min=-50, max=50)
        L1 = paddle.log(1 + paddle.exp(-s * inner_product)).mean()

        #do sgn
        b = feature.sign()
        inner_product_ = paddle.matmul(b, b, transpose_y=True) * 0.5
        sigmoid = paddle.nn.Sigmoid()
        L2 = (sigmoid(inner_product) - sigmoid(inner_product_)).pow(2).mean()

        return L1 + self._lambda * L2

#2019: if the result is OK, recommended
#no num_train, storage efficient, easy to understand
#good performance, easy ti reimplement
class DSHSDLoss2(paddle.nn.Layer):
    """
    # DSHSD(IEEE ACCESS 2019)
    # paper [Deep Supervised Hashing Based on Stable Distribution](https://ieeexplore.ieee.org/document/8648432/)
    # [DSHSDSD] epoch:70,  bit:48,  dataset:cifar10-1, MAP:0.809, Best MAP: 0.809
    # [DSHSDSD] epoch:250, bit:48, dataset:nuswide_21, MAP:0.809, Best MAP: 0.815
    # [DSHSDSD] epoch:135, bit:48, dataset:imagenet, MAP:0.647, Best MAP: 0.647
    """
    def __init__(self, n_class, bit, alpha, multi_label=False):
        super(DSHSDLoss2, self).__init__()
        self.m = 2 * bit     
        self.fc = paddle.nn.Linear(bit, n_class, bias_attr=False)  #use fc to connect bit and n_class

        # import numpy as np
        # tmp = np.load("a.npy")
        # print(tmp.shape)
        # for i in range(bit):
        #     for j in range(n_class):
        #         self.fc.weight[i, j] = float(tmp[j, i])

        self.alpha = alpha
        self.multi_label = multi_label
        print(self.fc.weight)

    def forward(self, feature, label):
        """
        for a batch result:
        feature: features
        label: labels
        index: image index in all the train dataset
        """
        feature = feature.tanh().astype("float32")
        dist = paddle.sum(
                    paddle.square((paddle.unsqueeze(feature, 1) - paddle.unsqueeze(feature, 0))), 
                    axis=2)
        
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

      def compare_dsh():
    """
    compare paddle and pytorch diff
    """
    from DSH import DSHLoss
    import numpy.random as random
    import torch
    import numpy as np

    num_train = 5000
    bit = 48
    n_class = 10
    alpha = 0.01   #best [0.001, 0.01]

    config = dict()
    config["num_train"] = num_train
    config["n_class"]   = n_class
    config["alpha"]     = alpha
    config["device"] = torch.device("cuda:0")

    criterion1 = DSHLoss(config, bit)
    criterion2 = DSHLoss2(num_train, n_class, bit, alpha)

    #generate input
    feature = random.randn(64, bit)
    label   = np.eye(n_class)[random.randint(0, 10, 64)]    
    index   = random.randint(0, 5000, 64)

    #loss1
    loss1 = criterion1(torch.from_numpy(feature).float().cuda(), 
                       torch.from_numpy(label).float().cuda(), 
                       torch.from_numpy(index).long().cuda(), config)

    #for paddle
    feature = paddle.to_tensor(feature)
    label = paddle.to_tensor(label)
    index = paddle.to_tensor(index)
    loss2 = criterion2(feature, label, index)

    print("loss1: {}".format(loss1))   
    print("loss2: {}".format(loss2))

    return

def compare_dhn():
    """
    compare paddle and pytorch diff
    """
    from DHN import DHNLoss
    import numpy.random as random
    import torch
    import numpy as np

    num_train = 5000
    bit = 48
    n_class = 10
    alpha = 0.1

    config = dict()
    config["num_train"] = num_train
    config["n_class"]   = n_class
    config["alpha"]     = alpha
    config["device"] = torch.device("cuda:0")

    criterion1 = DHNLoss(config, bit)
    criterion2 = DHNLoss2(num_train, n_class, bit, alpha)

    #generate input
    feature = random.randn(64, bit)
    label   = np.eye(n_class)[random.randint(0, 10, 64)]    
    index   = random.randint(0, 5000, 64)

    #loss1
    loss1 = criterion1(torch.from_numpy(feature).float().cuda(), 
                       torch.from_numpy(label).float().cuda(), 
                       torch.from_numpy(index).long().cuda(), config)

    #for paddle
    feature = paddle.to_tensor(feature)
    label = paddle.to_tensor(label)
    index = paddle.to_tensor(index)
    loss2 = criterion2(feature, label, index)

    print("loss1: {}".format(loss1))   
    print("loss2: {}".format(loss2))

    return

def compare_dpsh():
    """
    compare paddle and pytorch diff
    """
    from DPSH import DPSHLoss
    import numpy.random as random
    import torch
    import numpy as np

    num_train = 5000
    bit = 48
    n_class = 10
    alpha = 0.1

    config = dict()
    config["num_train"] = num_train
    config["n_class"]   = n_class
    config["alpha"]     = alpha
    config["device"] = torch.device("cuda:0")

    criterion1 = DPSHLoss(config, bit)
    criterion2 = DPSHLoss2(num_train, n_class, bit, alpha)

    #generate input
    feature = random.randn(64, bit)
    label   = np.eye(n_class)[random.randint(0, 10, 64)]    
    index   = random.randint(0, 5000, 64)

    #loss1
    loss1 = criterion1(torch.from_numpy(feature).float().cuda(), 
                       torch.from_numpy(label).float().cuda(), 
                       torch.from_numpy(index).long().cuda(), config)

    #for paddle
    feature = paddle.to_tensor(feature)
    label = paddle.to_tensor(label)
    index = paddle.to_tensor(index)
    loss2 = criterion2(feature, label, index)

    print("loss1: {}".format(loss1))   
    print("loss2: {}".format(loss2))

    return

def compare_dtsh():
    """
    compare paddle and pytorch diff
    """
    from DTSH import DTSHLoss
    import numpy.random as random
    import torch
    import numpy as np

    num_train = 5000
    bit = 48
    n_class = 10
    alpha = 0.1
    _lambda = 0.5

    config = dict()
    config["num_train"] = num_train
    config["n_class"]   = n_class
    config["alpha"]     = alpha
    config["lambda"]    = _lambda
    config["device"] = torch.device("cuda:0")

    criterion1 = DTSHLoss(config, bit)
    criterion2 = DTSHLoss2(alpha, _lambda)

    #generate input
    feature = random.randn(64, bit)
    label   = np.eye(n_class)[random.randint(0, 10, 64)]    
    index   = random.randint(0, 5000, 64)

    #loss1
    loss1 = criterion1(torch.from_numpy(feature).float().cuda(), 
                       torch.from_numpy(label).float().cuda(), 
                       torch.from_numpy(index).long().cuda(), config)

    #for paddle
    feature = paddle.to_tensor(feature)
    label = paddle.to_tensor(label)
    index = paddle.to_tensor(index)
    loss2 = criterion2(feature, label, index)

    print("loss1: {}".format(loss1))   
    print("loss2: {}".format(loss2))

    return

def compare_hashnet():
    """
    compare paddle and pytorch diff
    """
    from HashNet import HashNetLoss
    import numpy.random as random
    import torch
    import numpy as np

    num_train = 5000
    bit = 48
    n_class = 10
    alpha = 0.1

    config = dict()
    config["num_train"] = num_train
    config["n_class"]   = n_class
    config["alpha"]     = alpha
    config["device"] = torch.device("cuda:0")

    criterion1 = HashNetLoss(config, bit)
    criterion2 = HashNetLoss2(num_train, n_class, bit, alpha)

    #generate input
    feature = random.randn(64, bit)
    label   = np.eye(n_class)[random.randint(0, 10, 64)]    
    index   = random.randint(0, 5000, 64)

    #loss1
    loss1 = criterion1(torch.from_numpy(feature).float().cuda(), 
                       torch.from_numpy(label).float().cuda(), 
                       torch.from_numpy(index).long().cuda(), config)

    #for paddle
    feature = paddle.to_tensor(feature)
    label = paddle.to_tensor(label)
    index = paddle.to_tensor(index)
    loss2 = criterion2(feature, label, index)

    print("loss1: {}".format(loss1))   
    print("loss2: {}".format(loss2))

    return

if __name__=="__main__":
    
    #compare_dsh()
    #compare_dhn()
    #compare_dpsh()
    #compare_dtsh()
    compare_hashnet()
