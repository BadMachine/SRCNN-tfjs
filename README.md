# TensorflowJS implementation of SRCNN
Deep Convolutional Network for Image Super-Resolution implemented with Tensorflow.js

The original paper is [Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.00092)
<p align="center">
  <img src="https://github.com/BadMachine/SRCNN-tfjs/blob/master/additional/arch.jpg" width="800"/>
</p>

This implementation have some difference with the original paper, include:

* use Adam alghorithm for optimization, with learning rate 0.0003 for all layers.
* Use the opencv library to produce the training data and test data, not the matlab library. This difference may caused some deteriorate on the final results.
* I did not set different learning rate in different layer, but I found this network still work.
* The color space of YCrCb in Matlab and OpenCV also have some difference. So if you want to compare your results with some academic paper, you may want to use the code written with matlab.



### Data preparation
#### First of all you need to create two folders with training images and testing images. Then easily call:
```js
const cnn = require('srcnn');

let srcnn = new cnn();
```
```js
srcnn.prepare.prepare_data(path_to_test_images);
srcnn.prepare.prepare_crop_data(path_to_train_images);

```

### Training:
```js
srcnn.training.train({epochs: 300, batchSize: 128});
```


### Evaluating result:

##### Predicting on test data

```js
srcnn.prediction.testprediction(path_to_test_image);
```

##### Predicting on your pictures
```js
srcnn.prediction.predict_on_image(Path_to_image);
```
