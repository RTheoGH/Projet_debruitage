# PROJET DEBRUITAGE - M2 IMAGINE

## REY Emilien - REYNIER Théo

Ce projet a pour but de débruiter des images en utilisant un réseau de neurones (CNN). Le projet comporte notamment un encodeur, un décodeur ainsi qu'un GAN pour produire des images débruitées.

### Installation
```bash
pip install -r requirements.txt
```

### Initialisation

Dans la racine, créé un dossier `allImages` de telles sortes pour stockés les futures images bruités :
```
- allImages
    - train
        - truth
    - validation
        - truth
```

Ajouter des images dans les sous-dossiers de `allImages` :
```bash
cd Code
python move128.py
```

Bruiter les images (au choix) :
```bash
cd Code

python gaussian.py
python saltandpepper.py
python poisson.py
python periodic.py
```

### Entraînement

Pour entrainer le modèle :
```bash
cd Code
python unet_runner.py
```

Vous pouvez changé le bruit sur lequel s'entrainer (en changeant les dossiers traités):
```py
TRAIN_INPUT_DIR = '../allImages/train/noised/gaussian'  
TRAIN_GT_DIR =    '../allImages/train/truth' 
VAL_INPUT_DIR =   '../allImages/validation/noised/gaussian'  
VAL_GT_DIR =      '../allImages/validation/truth'
TEST_INPUT_DIR =  '../allImages/validation/noised/gaussian/test'
TEST_GT_DIR =     '../allImages/validation/truth/test'
```
Augmenter/diminuer le batch_size, le nombre d'epochs et d'autres paramètres :
```py
PRETRAIN_EPOCHS = 18          
L1_BASE = 80.0                
L1_FINAL = 30.0               
LAMBDA_PERCEPTUAL = 0.01      
LAMBDA_FM = 0.005           
LAMBDA_ADV = 0.005
LR = 0.00015                  
LR_D = 0.00005              
L1_LAMBDA = 25.0          

RESIDUAL_LEARNING = True
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
NUM_EPOCHS = 25
SAVE_EVERY = 1
NUM_WORKERS = 4
```

Une fois le modèle lancé, vous obtiendrez des résultats avec les images de bases, les images bruités, les images prédites, les PSNR entre les images de base et les images prédites, les SSIM (Structural Similarity Index Measure) et les DISTS (Deep Image Structure And Texture Similarity).

![png1](/readme/gt.png "gt") | ![png2](/readme/input.png "input") | ![png2](/readme/pred.png "pred")
:-------------------------:|:-------------------------:|:-------------------------:
![png3](/readme/psnr_curve.png "psnr")  |  ![png4](/readme/ssim_curve.png "ssim") | ![png2](/readme/dists_curve.png "dists")

### Test

Pour pouvez ensuite essayer de débruiter une image avec :
```bash
cd Code
python debruitage.py chemin_image.png
```

### Références

- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), vaste dataset d'images $32 times 32$.
- [DnCNN](https://arxiv.org/abs/1608.03981), Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
- [U-NET](https://stanford.edu/class/ee367/Winter2019/dua_report.pdf), Image Denoising Using a U-net 
- [U-NET](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9360532), A Residual Dense U-Net Neural Network for Image Denoising 
- [Poivre et Sel](https://fr.wikipedia.org/wiki/Bruit_poivre_et_sel), type de bruit dans une image.
- [Gaussien](https://fr.wikipedia.org/wiki/Bruit_gaussien), type de bruit dans une image.
- [Poisson](https://fr.wikipedia.org/wiki/Bruit_de_grenaille), type de bruit dans une image.
- [GAN](https://openaccess.thecvf.com/content/ACCV2020/papers/Tran_GAN-based_Noise_Model_for_Denoising_Real_Images_ACCV_2020_paper.pdf), GAN-based Noise Model for Denoising Real Images.