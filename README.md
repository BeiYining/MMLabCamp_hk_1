### OpenMMLab_homework_1

#### About This Work 
- Dataset: Flower 
https://pan.baidu.com/s/1RJmAoxCD_aNPyTRX6w97xQ passwd: 9x5u

- Model: MobileNet-v2


|    Model     | Params(M) | Flops(G) | Top-1 (%) |   
| :----------: | :-------: | :------: | :-------: | 
| MobileNet V2 |    2.23    | 0.32 |   97.0070   |  
#### Step
1. The data set was divided into training and validation sub-data sets in the ratio of 8:2, and the data set was sorted into the format of ImageNet.

2. Put the training and validation subsets under the train and val folders.

3. Create and edit the annotation file to write all class names into 'classes.txt', with each line representing one category.

4. Create and edit the annotation file to write all class names into 'classes.txt', with each line representing one category. Generate a list of training and validation subset annotations' train.txt 'and' val.txt '. Each line should contain a file name and its corresponding label.

5. run configs

#### how to run 
1. data 
```
python3 ./script/flower_split_ratio8-2.py
```
2. train
```
python tools/train.py   /home/yangshuo/past_comp/MMLabCamp_hk_1/configs/mobilenet-v2_flower.py  --gpus 2
```
It is worth mentioning that I set up two GPUs for training, but only card 0 actually participated in the training.
#### Other
The script given here is based on a teammate's(@Oscillated) code, which I modified.