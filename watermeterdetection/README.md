# Water Meter Detection
## Background
Instrument reading detection is very important 
## Requirements
* Neural Network based model
* Python and PyTorch are required in the experiment
* There is GPU RTX 2080 Ti used for data training. At test time, the algorithm should work without GPU.
## Datasets
We used the Toloka WaterMeters dataset collected by Roman Kucev from TrainingData.ru, contains 1244 images of hot and cold water meters as well as their readings and coordinates of the displays showing those readings. Each image contains exactly one water meter. The archive also includes the pictures of the results of segmentation with the masks and collages. Toloka was used for photo capturing, segmentation, and recognizing the readings.

The data could be downloaded [here](https://toloka.ai/datasets/).

Another dataset that we found but not yet trained on consists of both digital meter and pointer meters' reading images. The data could be downloaded [here](https://aistudio.baidu.com/aistudio/datasetdetail/157981).

To preprocess the dataset, we applied the normalization to stitch with the ResNet normalization on real-world images.

## About model and experiment
For detailed code see WaterMeters_detection.ipynb.

## Reference
* [Fasterrcnn_resnet50_fpn](https://aistudio.baidu.com/aistudio/datasetdetail/157981)

