## Learning Selfie-Friendly Abstraction from Artistic Style Images

Yicun Liu | Jimmy Ren |  Jianbo Liu |  Jiawei Zhang | Xiaohao Chen     

#### *ACML 2018*    

---

This repository contains code for the paper: [Learning Selfie-Friendly Abstraction from Artistic Style Images](http://proceedings.mlr.press/v95/liu18a.html).

Contact: Yicun Liu (stanleylau@link.cuhk.edu.hk)

<img src="pics/large.jpg" width="1200" />

#### Prerequisites   
The code is tested on 64 bit Linux (Ubuntu 14.04 LTS). You should also install Matlab (We have tested on R2015a). We have tested our code on GTX TitanX (Maxwell) with CUDA8.0+cuDNNv5. Please install all these prerequisites before running our code.

#### Installation
1. Clone the code. 
   ```Shell  
   git clone https://github.com/DandilionLau/Selfie-Friendly-Abstraction.git 
   cd Selfie-Friendly-Abstraction  
   ```

2. Build standard caffe follow the [instruction](http://caffe.berkeleyvision.org/installation.html).    
   ```Shell
   cd caffe/
   # Modify Makefile.config according to your Caffe installation. 
   # Remember to allow CUDA and CUDNN.
   make -j12
   make matcaffe
   ```

3. Cutomize caffe.    

   Add the following message to  `caffe.proto`  to configure the new `NNUp Upsample Layer`:   

   ```Proto
   Message Parameter{
   // Insert to existing class. Try not to conflict with any existing message numbers.
     optional NNUpsampleParameter nn_upsample_param = 163;
   }
   message NNUpsampleParameter {
   // Append to the last of caffe.proto file
     optional uint32 resize = 1 [default = 2];
   }
   ```
   Copy `include/nn_upsample_layer.hpp` to `caffe/include/caffe/layers/`. Copy `src/nn_upsample_layer.cpp` and `src/nn_upsample_layer.cu` to `caffe/src/caffe/layers/`. Then recompile both caffe and matcaffe.   
   ```Shell
   make -j12
   make matcaffe
   ```

4. Download training models.   
   To prepare for the testing step, you may simply download the trained caffemodels from [[DropBox]](https://www.dropbox.com/s/1md2kewhmnhg6kl/style.zip?dl=0)[[BaiduYun]](https://pan.baidu.com/s/1mWnx6EyA1WuEUUZbfJu96g) and put them to the `model/style/` directory .    

   Additionally, if you want to train your own style abstraction model, you need to download the VGG-16 model from [[VGG Website]](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel)[[DropBox]](https://www.dropbox.com/s/jwhjdqlg2g6y0bb/vgg16.caffemodel?dl=0)[[BaiduYun]](https://pan.baidu.com/s/1SWUx-7siOjsL-KlhDbyKTw) to compute the perceptual loss. It should be put to the `model/vgg_16layers/` directory.

#### Training
1. Generate image patches   
   Run `data/GenPatches_train_6chs.m` and ``data/GenPatches_val_6chs.m`` at MATLAB to extract image patches for training and validation. We provide 40 selfie images and their corresponding output images generated from Prisma. The selfie image and stylistic reference image directory is at `data/training/`. You may replace it with your own dataset to train with different styles.
2. Training the model   
   Run `train/train_6chs_reshape.m` at MATLAB to train the model. Remember to include matcaffe before training. In our experiment, the balance factor between loss_pixel and loss_feat is set as 1000. 

#### Testing 
1. Run `test/test_6chs_reshape.m` at MATLAB to test the model. Remember to include matcaffe before runing the test. We provide 99 images from Flickr for testing, including portraits, landscapes, wild lifes and other scenes. The image directory is at `data/testing/`. You may replace it with your own dataset for testing.
2. For inter-frame consitency test, please visit our [online demo](https://youtu.be/0AsY26MHih4) to check the results.

#### Results
<img src="pics/people.jpg" width="1200" />
<p></p>
<img src="pics/scene.jpg" width="1200" />  

#### Citation
   Please cite our paper if you find it helpful for your work:   
```
@InProceedings{pmlr-v95-liu18a,
title = {Learning Selfie-Friendly Abstraction from Artistic Style Images},
author = {Liu, Yicun and Ren, Jimmy and Liu, Jianbo and Chen, Xiaohao},
booktitle = {Proceedings of The 10th Asian Conference on Machine Learning},
year = {2018}
}
```
```
@article{liu2018learning,
  title={Learning Selfie-Friendly Abstraction from Artistic Style Images},
  author={Liu, Yicun and Ren, Jimmy and Liu, Jianbo and Zhang, Jiawei and Chen, Xiaohao},
  journal={arXiv preprint arXiv:1805.02085},
  year={2018}
}
```


