require('dotenv').config();
//const tf = require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs-node-gpu');
const model_to_predict = require("./SRCNN")
const cv = require("opencv4nodejs")


function slice_to_predict(array){
	let output = [];

		array.forEach(function (first_layer) {
			output.push([]);
			first_layer.forEach(function (second_layer) {
				output[output.length - 1].push(second_layer[0]);
			});
		});
	return output;
}



function to4dtensor(array){
	let output = [];
	//output.push([]);
	for (let it = 0;it< array.length;it++){
		output.push([]);
		let temporary=[];
		let elem_length = array[it].length;
		for(let that = 0; that< elem_length; that++){
			temporary.push([array[it][that][0]])
			//console.log(array[it][that])
		}
		output[it].push(temporary);
	}
	//console.log(output[0][0][0][0]);
	return output;
	
}




function merge_resized_with_original(org, resized){
	let original = org.getDataAsArray();
	let res = resized.getDataAsArray();
	
	for(let it = 0; it< original.length; it++){
		let temp = original[it].length;
		for(let that = 0; that< temp; that++){
			//console.log(original[it][that])
			//console.log(res[it][that])
			original[it][that][0] = res[it][that];
		}
		
	}
	
	return original;
}





function toResize(Mat){
	let array = Mat.getDataAsArray();
	let out = [];
	for(let it = 0; it<array.length; it++){
		let temporary = [];
		for(let that =0; that<array[it].length; that++){
			temporary.push(array[it][that][0])
			
		}
		out.push(temporary);
	}
	return out;
}





function fixValues(array){

	for(let firstLayer = 0; firstLayer< array.length; firstLayer++){
		for(let secondLayer = 0; secondLayer< array[firstLayer].length; secondLayer++)
			
			for(let thirdLayer = 0; thirdLayer< array[firstLayer][secondLayer].length; thirdLayer++){
				if (array[firstLayer][secondLayer][thirdLayer][0] > 255){
					array[firstLayer][secondLayer][thirdLayer][0] = 255;
				} 
				if(array[firstLayer][secondLayer][thirdLayer][0] < 0){
					array[firstLayer][secondLayer][thirdLayer][0] = 0;
				}
			}
			
	}
	return array;
}




function mat_from4dtensor(array){
	 let output = [];
	 let first_layer = array[0];
	 for(let it = 0; it < first_layer.length; it++){
		 let layer_length = first_layer[it].length;
		 let temporary = [];
		 for(let that = 0; that< layer_length; that++){
			 temporary.push(first_layer[it][that][0])
		 }
		 output.push(temporary)
	 }
	 return output;
 }



function restoreColor(original, predicted){
	
	for (let it = 6; it< original.length-6; it++){
		for(let that = 6; that< original[it].length-6; that++){
			original[it][that][0] = predicted[it-6][that-6];
			//console.log(original[it][that])
		}
		
	}
	return original;
}


 module.exports.predict = async function(){
	 
	let srcnn_model = model_to_predict.predict_model();


	const model = await tf.loadLayersModel(
		'file://./tmp/my-model/best/model.json');
		
		srcnn_model.layers[0].setWeights(model.layers[0].getWeights())
		srcnn_model.layers[1].setWeights(model.layers[1].getWeights())
		srcnn_model.layers[2].setWeights(model.layers[2].getWeights())

	
	let IMG_NAME = "./Data/Set14/monarch.bmp"
	let INPUT_NAME = "input2.jpg"
	let OUTPUT_NAME = "pre2.jpg"
	
    let img = cv.imread(IMG_NAME);
	let original_image = img;
		img = img.cvtColor(cv.COLOR_BGR2YCrCb);
	
	let shape = [img.cols, img.rows];
		
	let to_resize_img = toResize(img);
		to_resize_img = new cv.Mat(to_resize_img, cv.CV_8UC1);

	let	Y_img = to_resize_img.resize(~~(shape[1] / 2), ~~(shape[0] / 2), cv.INTER_CUBIC);
		Y_img = Y_img.resize(shape[1], shape[0], cv.INTER_CUBIC);

	let Y_img_merged = merge_resized_with_original(img, Y_img);
	
	let colorized = Y_img_merged;
		Y_img_merged = tf.div(tf.tensor(Y_img_merged), tf.scalar(255.0))
		Y_img_merged = Y_img_merged.arraySync();

	let	Y = to4dtensor(Y_img_merged);

		Y = tf.tensor4d(Y)
		Y = tf.transpose(Y,[1, 0, 2, 3]);

		let pre = srcnn_model.predict(Y, {batch_size:1});
			pre = tf.mul(pre, tf.scalar(255.0))
			pre = pre.arraySync()
		let fixed = fixValues(pre);
			fixed = tf.tensor(fixed).asType('int32');
			fixed = fixed.arraySync();
			fixed = mat_from4dtensor(fixed);
			colorized = new cv.Mat(colorized, cv.CV_8UC3);//.cvtColor(cv.COLOR_YCrCb2BGR);
			cv.imshow("Original", colorized.cvtColor(cv.COLOR_YCrCb2BGR))
			colorized = colorized.getDataAsArray();
		let tried = restoreColor(colorized, fixed);
			tried = new cv.Mat(tried, cv.CV_8UC3).cvtColor(cv.COLOR_YCrCb2BGR);
			cv.imshow("Predicted", tried)
			cv.waitKey(0)


 }
