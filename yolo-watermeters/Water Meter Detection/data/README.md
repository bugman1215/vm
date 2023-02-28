### Toloka Water Meters
The dataset consists of 1244 images of hot or cold water meters. Each image contains exactly one water meter.  The readings and the coordinates of digits are provided for each water meter image as well.
![](https://hsto.org/webt/pv/83/un/pv83unxagbpbdifzxexbfpuefwo.png)
The dataset is assembled by Kutsev Roman from the [TrainingData.ru](http://trainingdata.ru/)  company for an article on Habr.

### How the data was collected
The detailed process of collecting the dataset is described in the article  [“Create your own dataset for recognition of water meters on Yandex.Toloka”](https://habr.com/ru/company/ods/blog/469633/). The data.yaml file to sppecify where our training and validation data is the number of classes that we want to detect and the names corresponding to those classes. We apply data augmentation to the images , flipping and rotating them to obtain more data and to generate a better generalization and learning in the model.

### Archive contents

## data.tsv
- photo\_name -- photo name
- value -- reading
- location -- display coordinates

## data.yaml
-train: ../train/images
-val: ../valid/images
-nc: 1
-names: ['numbers']


## images
Directory with images of water meters

## masks
Directory with masks of water meters

## collage
Directory with collages of water meters