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
