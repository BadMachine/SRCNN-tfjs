
require('dotenv').config();
const tf = require('@tensorflow/tfjs-node-gpu');
const model = require("./SRCNN");
const fs = require("fs")
const preparation = require("./prepare_data");

const DATA_PATH = "../SRCNN/Set5/"
const TEST_PATH = "../SRCNN/Set14/"
 module.exports.train = async function(config){
	 
	if(config.createDataset){
	let training_data = preparation.prepare_crop_data(DATA_PATH);
	let prepared_test = preparation.prepare_data(TEST_PATH);
	}
	
	let srcnn_model = model.create_model();
		srcnn_model.summary();
	
	let to_train = fs.readFileSync('../dataset/croped_Data.json', 'utf8');
		to_train = tf.tensor(JSON.parse(to_train) );
		to_train = tf.transpose(to_train,[0, 2, 3, 1]);
		
	let to_train_sec = fs.readFileSync('../dataset/croped_Label.json', 'utf8');
		to_train_sec = tf.tensor(JSON.parse(to_train_sec) );
		to_train_sec = tf.transpose(to_train_sec,[0, 2, 3, 1]);
		
	let to_val = fs.readFileSync('../dataset/croped_validation_Data.json', 'utf8');
		to_val = tf.tensor(JSON.parse(to_val) );
		to_val = tf.transpose(to_val,[0, 2, 3, 1]);
		
	let to_val_sec = fs.readFileSync('../dataset/croped_validation_Label.json', 'utf8');
		to_val_sec = tf.tensor(JSON.parse(to_val_sec));
		to_val_sec = tf.transpose(to_val_sec,[0, 2, 3, 1]);	
		
		
	const bestModelPath = 'file://../tmp/my-model/best/best/'
	let bestLoss = 100;
		const history = await srcnn_model.fit(to_train, to_train_sec, {
			epochs: 300,
			batchSize: 128,
			shuffle: true,
			verbose: 2,
			validationData: [to_val, to_val_sec],
    callbacks:  {
		
		onEpochBegin :async function (epoch, logs) {
			//console.log(logs)
		},
        onEpochEnd: async function (epoch, logs) {
			tf.node.tensorBoard('../tmp/fit_logs_1');
            if (logs.loss < bestLoss) {
                console.log("Got better result, saving model to ", bestModelPath.slice(7, bestModelPath.slicelength))
                bestValLoss = logs.val_loss;
                await srcnn_model.save(bestModelPath);
			try{		
					let model = await tf.loadLayersModel(bestModelPath+"model.json");
					srcnn_model.layers[0].setWeights(model.layers[0].getWeights())
					srcnn_model.layers[1].setWeights(model.layers[1].getWeights())
					srcnn_model.layers[2].setWeights(model.layers[2].getWeights())
				
				console.log("model refreshed");
			}catch(e){console.log("No best model were found")};
            }
        },
        onTrainEnd: async function () {
			
        },
	
    }
		});
		
	
	await srcnn_model.save('file://../tmp/my-model/');
	srcnn_model.predict(to_train).print();
 }