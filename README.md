# TensorflowJS implementation of SRCNN
Deep Convolutional Network for Image Super-Resolution implemented with Tensorflow.js

The original paper is [Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.00092)
<p align="center">
  <img src="https://github.com/MarkPrecursor/SRCNN-keras/blob/master/SRCNN.png" width="800"/>
</p>

This implementation have some difference with the original paper, include:

* use Adam alghorithm for optimization, with learning rate 0.0003 for all layers.
* Use the opencv library to produce the training data and test data, not the matlab library. This difference may caused some deteriorate on the final results.
* I did not set different learning rate in different layer, but I found this network still work.
* The color space of YCrCb in Matlab and OpenCV also have some difference. So if you want to compare your results with some academic paper, you may want to use the code written with matlab.


### Evaluating result:

##### Predicting on test data
```js
prediction.predict();
```

##### Predicting on your pictures
```js
predict.predict_on_image(Path_to_image);
```
### Training:
```js
training.train({ createDataset: false }); // use true if you want to create or refresh dataset
```
