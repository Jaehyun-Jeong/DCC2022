# VGG16

## 01

dataset: train_val_test_dataset (그냥 6:2:2로 나눈 데이터)<br/>
batch_size = 128<br/>
lr = 0.001<br/>
처음에 돌렸을 때 test loss가 변경되지 않아 지금의 learning rate를 줄일 필요가 있음을 의미한다.</br>

## 02

dataset: train_val_test_dataset</br>
batch_size = 128</br>
lr = 0.01</br>
stepLR, gamma = 0.1, step_size = 100</br>

Train_loss을 출력해서 gd가 이뤄지고 있는지까지 확인하고자 했는데 결과,  2.35에서 줄어들지 않음을 알 수 있고, 이는 learning rate를 줄이게 되었다.

## 03

dataset: train_val_test_dataset<br/>
batch_size = 128<br/>
lr = 1e-6<br/>

## 04

dataset: augmented_dataset (affine과 rotation만 적용하여 밸런스를 맞춘 데이터 셋)<br/>
batch_size = 256<br/>
lr = 1e-6<br/>

Batch_size가 늘어났을 때의 경향을 보기 위해 256으로 늘렸다.

## 05

dataset: augmented_dataset<br/>
batch_size = 256<br/>
lr = 1e-6<br/>

## 06

dataset: random_augmented_dataset (rotation만 적용한 데이터 셋은 다양하지 않다고 판단하여, 6가지 augmentation을 랜덤하게 사용하여 만든 데이터 셋, 최대 원래 데이터의 7배까지 augmentation), rotation을 적용하는 것보다는 허용범ㅁ위가 넓어짐<br/>
batch_size = 128<br/>
lr = 1e-6<br/>

전의 augmentation dataset은 affine 하나와 rotation으로만 이루어져 있다. 하지만 좀 더 다양한 augmentation을 적용시킴으로써 general한 augmentation을 하기 위해 시도했다.

그리고 batch_size를 다시 낮춘 이유는 lr이 낮을 때, batch_size도 낮은 값이 좋다는 결과를 (예지가 발견, https://www.sciencedirect.com/science/article/pii/S2405959519303455#fig2 <- 이 논문 리뷰 보다가 발견) 발견했기 때문이다.

## 07

dataset: random_augmented_dataset_v2 (random_augmented_dataset에서 augmentation 배수를 12배로 증가시킨 데이터 셋)<br/>
batch_size = 128<br/>
lr = 1e-6<br/>

Random_augmented_dataset은 최대 7배까지만 데이터를 증가시킨다. 하지만 이렇게 해도 데이터가 너무 적은 클래스가 존재하여, 최대 12배까지 늘린 random_augmented_dataset_v2를 만들었다.

## 08

focal_loss 사용<br/>
dataset: random_augented_dataset_v2<br/>
batch_size = 128<br/>
lr = 1e-6<br/>

Focal loss를 사용하여 07과 같은 테스트를 진행

## 09

focal_loss 사용<br/>
dataset: random_augmented_dataset_v2<br/>
batch_size = 128<br/>
lr = 1e-6<br/>

![](./static/VGG16_test_09_train_test_loss.png)<br/>
![](./static/VGG16_test_09_f1score.png)<br/>

## 10

focal_loss 사용<br/>
dataset: random_augmented_dataset_v3 (L2_25, L2_30을 80%까지 줄이고 random_augmented_dataset과 동일한 코드로 augmentation한 데이터 셋 (최대 7배))<br/>
batch_size = 128<br/>
lr = 1e-6<br/>

![](./static/VGG16_test_10_train_test_loss.png)<br/>
![](./static/VGG16_test_10_f1score.png)<br/>

test_10부터 lr 3e-6에서 30epoch, 2e-6에서 50epoch, 1e-6에서 100epoch<br/>

> test_loss가 떨어졌음에도 불구하고 f1-score가 올라간 경우가 발견되었다. 이는 예측의 확신은 떨어졌지만 결국 예측을 잘하게 된 경우라고 생각된다.

## 11

focal_loss 사용<br/>
dataset: random_augmented_dataset_v4 (L2_25, L2_30을 60%까지 줄이고 random_augmented_dataset과 동일한 코드로 augmentation한 데이터 셋 (최대 7배))<br/>
batch_size = 128<br/>
lr = 1e-6<br/>

random_augmented_dataset_v3가 더 좋은 데이터 셋이라고 판단, 따라서 이제부터 random_augmented_dataset_v3로 실험을 진행

## 12

focal_loss 사용<br/>
dataset: random_augmented_dataset_v3 (L2_25, L2_30을 60%까지 줄이고 random_augmented_dataset과 동일한 코드로 augmentation한 데이터 셋 (최대 7배))<br/>
batch_size = 128<br/>
lr = 1e-6<br/>

아래의 주소에서 소개된 WeightedRandomSampler를 사용하여 테스트 진행<br/>
<https://www.kaggle.com/code/maulberto3/mnist-simple-pytorch-cnn-3-layer-98/notebook?scriptVersionId=87318629&cellId=6><br/>

> WeightedRandomSampler: imbalance 데이터에서 batch 데이터를 가져올 때, balance하게 가져오기 위한 샘플링 방법

# ResNet18

## 01

dataset: augmented_dataset<br/>
batch_size = 256<br/>
lr = 0.0003<br/>
stepLR, gamma = 0.1, step_size = 30, last_epoch = 30 * 2<br/>

![](./static/ResNet18_test_01_train_test_loss.png)<br/>
![](./static/ResNet18_test_01_f1score.png)<br/>

## 02

dataset: random_augmented_dataset<br/>
batch_size = 256<br/>
lr = 1e-6<br/>

![](./static/ResNet18_test_02_train_test_loss.png)<br/>
![](./static/ResNet18_test_02_f1score.png)<br/>

# ResNet50

torchvision의 소스코드를 참고하여 ResNet50 class를 만들었다.<br/>
<https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/resnet.py#L124>

## 01

focal_loss 사용<br/>
dataset: random_augmented_dataset_v3 (L2_25, L2_30을 80%까지 줄이고 random_augmented_dataset과 동일한 코드로 augmentation한 데이터 셋 (최대 7배))<br/>
batch_size = 128<br/>
lr = 1e-6<br/>
