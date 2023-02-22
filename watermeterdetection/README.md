# Water Meter Detection
## Background
Nowadays, industrial instruments are becoming increasingly functional and play an increasingly important role in the industry. With the continuous integration and deepening of electronic information and modern industry, industrial production is gradually developing towards automation and intelligence. IoT-based instrument reading is widely used, but this method is only applicable to industrial instruments with communication interfaces for reading. In the current industrial life, there are still many traditional industrial meters without communication interfaces, which are mostly read manually. However, the manual reading of the meter is a large and inefficient workload, which can easily cause visual fatigue and thus lead to errors in the values read.

Industrial meters at this stage can be divided into pointer meters and digital meters according to the type of dial. The experiment mainly used water meter readings, which is a category of digital meters, and was able to obtain digitaal readings from the original images using a neural network based method.
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

