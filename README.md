# DeepHash-Paddle
Implementation of Some Deep Hash Algorithms Baseline with PaddlePaddle

# How to run
My environment is
```
python==3.7.0  paddle==2.2.1  
```

You can easily train and test any algorithm just by
```
pyhon DSH.py  
pyhon DPSH.py  
pyhon DHN.py    
pyhon DSDH.py    
``` 

# Precision Recall Curve
<img src="https://github.com/Intsigstephon/DeepHash-Paddle/blob/main/utils/pr.png"  alt="Precision Recall Curve"/><br/>  
I add some code in DSH.py:
```
if "cifar10-1" == config["dataset"] and epoch > 29:
    P, R = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy())
    print(f'Precision Recall Curve data:\n"DSH":[{P},{R}],')
```
To get the Precision Recall Curve, you should copy the data (which is generated by the above code ) to precision_recall_curve.py  and run this file.  
```
cd utils
pyhon precision_recall_curve.py   
```
# Dataset
There are three different configurations for cifar10  
- config["dataset"]="cifar10" will use 1000 images (100 images per class) as the query set, 5000 images( 500 images per class) as training set , the remaining 54,000 images are used as database.
- config["dataset"]="cifar10-1" will use 1000 images (100 images per class) as the query set, the remaining 59,000 images are used as database, 5000 images( 500 images per class) are randomly sampled from the database as training set.  
- config["dataset"]="cifar10-2" will use 10000 images (1000 images per class) as the query set, 50000 images( 5000 images per class) as training set and database.


You can download   NUS-WIDE [here](https://github.com/TreezzZ/DSDH_PyTorch)     
Use data/nus-wide/code.py to randomly select 100 images per class as the query set (2,100 images in total). The remaining images are
used as the database set, from which we randomly sample 500 images per class as the training set (10, 500 images
in total).

You can download  ImageNet, NUS-WIDE-m and COCO dataset [here](https://github.com/thuml/HashNet/tree/master/pytorch) where is the data split  copy from,  or [Baidu Pan(Password: hash)](https://pan.baidu.com/s/1_BiOmeCRYx6cVTWeWq-O9g).
  
NUS-WIDE-m is different from  NUS-WIDE, so i made a distinction.  

269,648 images in NUS-WIDE , and 195834 images which are associated with 21 most frequent concepts.     

NUS-WIDE-m has 223,496 images,and  NUS-WIDE-m  is used in [HashNet(ICCV2017)](http://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf) and code [HashNet caffe and pytorch](https://github.com/thuml/HashNet)    

download [mirflickr](https://press.liacs.nl/mirflickr/mirdownload.html) , and use ./data/mirflickr/code.py to randomly select 1000 images as the test query set and 4000 images as the train set.
 
# Paper And Code
It is difficult to implement all by myself, so I made some modifications based on these codes  
DSH(CVPR2016)  
paper [Deep Supervised Hashing for Fast Image Retrieval](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf)  
code [DSH-pytorch](https://github.com/weixu000/DSH-pytorch)

DPSH(IJCAI2016)  
paper [Feature Learning based Deep Supervised Hashing with Pairwise Labels](https://cs.nju.edu.cn/lwj/paper/IJCAI16_DPSH.pdf)   
code [DPSH-pytorch](https://github.com/jiangqy/DPSH-pytorch)

DHN(AAAI2016)  
paper [Deep Hashing Network for Efficient Similarity Retrieval](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-hashing-network-aaai16.pdf)  
code [DeepHash-tensorflow](https://github.com/thulab/DeepHash) 

HashNet(ICCV2017)  
paper [HashNet: Deep Learning to Hash by Continuation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf)  
code [HashNet caffe and pytorch](https://github.com/thuml/HashNet)

GreedyHash(NIPS2018)  
paper [Greedy Hash: Towards Fast Optimization for Accurate Hash Coding in CNN](https://papers.nips.cc/paper/7360-greedy-hash-towards-fast-optimization-for-accurate-hash-coding-in-cnn.pdf)  
code [GreedyHash](https://github.com/ssppp/GreedyHash) 

DSDH(NIPS2017)  
paper [Deep Supervised Discrete Hashing](https://papers.nips.cc/paper/6842-deep-supervised-discrete-hashing.pdf)  
code [DSDH_PyTorch](https://github.com/TreezzZ/DSDH_PyTorch)

LCDSH(IJCAI2017)  
paper [Locality-Constrained Deep Supervised Hashing for Image Retrieval](https://www.ijcai.org/Proceedings/2017/0499.pdf)

DSHSD(IEEE ACCESS 2019)  
paper [Deep Supervised Hashing Based on Stable Distribution](https://ieeexplore.ieee.org/document/8648432/)

Deep Unsupervised Image Hashing by Maximizing Bit Entropy(AAAI2021)  
paper [Deep Unsupervised Image Hashing by Maximizing Bit Entropy](https://arxiv.org/pdf/2012.12334.pdf)  
code [Deep-Unsupervised-Image-Hashing](https://github.com/liyunqianggyn/Deep-Unsupervised-Image-Hashing)

# Mean Average Precision,48 bits[AlexNet].
<table>
    <tr>
        <td>Algorithms</td><td>dataset</td><td>this impl.</td><td>paper</td>
    </tr>
    <tr>
        <td >DSH</td><td >cifar10-1</td> <td >0.800</td> <td >0.6755</td>
    </tr>
    <tr>
        <td ></td><td >nus_wide_21</td> <td >0.798</td> <td >0.5621</td>
    </tr>
    <tr>
        <td ></td><td >ms coco</td> <td >0.655</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >imagenet</td> <td >0.576</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >mirflickr</td> <td >0.735</td> <td >-</td>
    </tr>
</table>

Due to time constraints, I can't test many hyper-parameters.
If you have any problems, feel free to contact me by email(451685052@qq.com) or raise an issue. 

