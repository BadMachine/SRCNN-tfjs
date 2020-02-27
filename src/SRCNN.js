require('dotenv').config();
const tf = require('@tensorflow/tfjs-node-gpu');

 module.exports.create_model = ()=>{
	 const SRCNN = tf.sequential();
	 SRCNN.add(tf.layers.conv2d({ filters: 128, kernelSize: [9, 9], kernelInitializer: 'glorotUniform', activation: 'relu', padding: 'valid', useBias: true, inputShape: [32, 32, 1] , name: "conv2d_1"}));
	 SRCNN.add(tf.layers.conv2d({ filters: 64, kernelSize: [3, 3], kernelInitializer: 'glorotUniform', activation: 'relu', padding: 'same', useBias: true , name: "conv2d_2"}));
	 SRCNN.add(tf.layers.conv2d({ filters: 1, kernelSize: [5, 5], kernelInitializer: 'glorotUniform', activation: 'linear', padding: 'valid', useBias: true , name: "conv2d_3" }));
	 SRCNN.compile({ optimizer: tf.train.adam(3.0e-4), loss: tf.losses.meanSquaredError, metrics: ['mse'] });
	 return SRCNN;
 }
 
 module.exports.predict_model = ()=>{
	const SRCNN = tf.sequential();
	SRCNN.add(tf.layers.conv2d({ filters: 128, kernelSize: [9, 9], kernelInitializer: 'glorotNormal', trainable: false, activation: 'relu', padding: 'valid', useBias: true, inputShape: [null, null, 1] }));
	SRCNN.add(tf.layers.conv2d({ filters: 64, kernelSize: [3, 3], kernelInitializer: 'glorotNormal', trainable: false, activation: 'relu', padding: 'same', useBias: true }));
	SRCNN.add(tf.layers.conv2d({ filters: 1, kernelSize: [5, 5], kernelInitializer: 'glorotNormal', trainable: false, activation: 'linear', padding: 'valid', useBias: true }));
	const adam = tf.train.adam(0.0003);
	SRCNN.compile({ optimizer: tf.train.adam(3.0e-4), loss: 'meanSquaredError', metrics: ['mse']});
	return SRCNN;
}