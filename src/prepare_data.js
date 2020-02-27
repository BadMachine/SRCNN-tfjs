
require('dotenv').config();
const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require("fs");
const cv = require("opencv4nodejs");
const BLOCK_STEP = 16;
const BLOCK_SIZE = 32;
const Random_Crop = 30;
const Patch_size = 32;
const label_size = 20;
const conv_side = 6;
const scale = 2;



function random_int(low, high, size) {
	let result = [];
	for (let it = 0; it < size; it++) {
		let rand = low + Math.random() * (high + 1 - low);
		result.push(Math.floor(rand));
	}
	return result;
}



function isNumber(n) {
  return !isNaN(parseFloat(n)) && isFinite(n);
}

function roundNumericValuesInArray(array, precision){
    var roundedArray = [];

    array.forEach(function round(elem){
        if(isNumber(elem)) {
            roundedArray.push(Number(elem.toFixed(precision)));
        } else if(elem.constructor === Array){
            roundedArray.push(roundNumericValuesInArray(elem, precision));
        } else {
            roundedArray.push(elem);
        }
    })

    return roundedArray;
}





function create_conv_center(length, array) {
	let output = [];

	for (let it = length; it < array.length - length; it++) {
		let temporary = []
		for (let that = length; that < array.length - length; that++) {
			temporary.push(array[it][that])
		}
		output.push(temporary);
	}

	return output;
}

module.exports.prepare_data = function (path) { 
   
let names = fs.readdirSync(path);
	names = names.sort();
let nums = names.length;

let data = [];
let label = [];

	for (let i = 0; i < nums; i++) {
		let name = path + names[i];
		let hr_img = cv.imread(name);
			console.log(name, " image read...");
		let shape = [hr_img.rows, hr_img.cols];
			hr_img = hr_img.cvtColor(cv.COLOR_BGR2YCrCb);
		let test = hr_img.getDataAsArray();
			hr_img = [];

		test.forEach(function (first_layer) {
			hr_img.push([]);
			first_layer.forEach(function (second_layer) {
				hr_img[hr_img.length - 1].push(second_layer[0]);
			});
		});

			hr_img = new cv.Mat(hr_img, cv.CV_8UC1);


		// two resize operation to produce training data and labels
		let lr_img = hr_img.resize(~~(shape[1] / scale), ~~(shape[0] / scale), cv.INTER_CUBIC);

			lr_img = lr_img.resize(shape[1], shape[0], cv.INTER_CUBIC);



		//produce Random_Crop random coordinate to crop training img
		let Points_x = random_int(0, Math.min(shape[0], shape[1]) - Patch_size, Random_Crop);

		let Points_y = random_int(0, Math.min(shape[0], shape[1]) - Patch_size, Random_Crop);


		for (let it = 0; it < Random_Crop; it++) {

			let lr_img_arr = lr_img.getDataAsArray();
			let hr_img_arr = hr_img.getDataAsArray();


			let lr_patch = new cv.Mat(lr_img_arr, cv.CV_32FC1);
				lr_patch = lr_patch.div(255.0);
				lr_patch = lr_patch.getRegion(new cv.Rect(Points_x[it], Points_y[it], Patch_size, Patch_size));


			let hr_patch = new cv.Mat(hr_img_arr, cv.CV_32FC1);
				hr_patch = hr_patch.div(255.0);
				hr_patch = hr_patch.getRegion(new cv.Rect(Points_x[it], Points_y[it], Patch_size, Patch_size));

			
				data.push([lr_patch.getDataAsArray()]);

			let hr_roi = new cv.Rect(0, 0, label_size, label_size);
			let hr = hr_patch.getRegion(hr_roi);  

				label.push([hr.getDataAsArray()]);

		}
		console.log(i," - image loaded to crop and prepare validation set function");
	}
	
	fs.writeFileSync('../dataset/croped_validation_Data.json', JSON.stringify(data), 'utf8');
	fs.writeFileSync('../dataset/croped_validation_Label.json', JSON.stringify(label), 'utf8');
	return [data, label];
	console.log(label[0][0][0])
	return [data, label];
};

module.exports.prepare_crop_data = function (path) { 
    
let names = fs.readdirSync(path);
	names = names.sort();
let nums = names.length;

let data = [];
let label = [];
let tensors_array = [];


for (let iterator = 0; iterator < nums; iterator++) {

	let name = path + names[iterator];
		
	let hr_img = cv.imread(name)
		console.log(name, " image read...");
	let shape = [hr_img.rows, hr_img.cols];
		hr_img = hr_img.cvtColor(cv.COLOR_BGR2YCrCb);
	let test = hr_img.getDataAsArray();
		hr_img = [];
		test.forEach(function (first_layer) {
			hr_img.push([]);
			first_layer.forEach(function (second_layer) {
				hr_img[hr_img.length - 1].push(second_layer[0]);
			});
		});
		//console.log(hr_img)
		hr_img = new cv.Mat(hr_img, cv.CV_8UC1);

		// two resize operation to produce training data and labels
	let lr_img = hr_img.resize(~~(shape[1] / scale), ~~(shape[0] / scale), cv.INTER_CUBIC);
		lr_img = lr_img.resize(shape[0], shape[1], cv.INTER_CUBIC);

	let hr_img_array = hr_img.getDataAsArray();
	let lr_img_array = lr_img.getDataAsArray();

	const width_num = parseInt((shape[0] - (BLOCK_SIZE - BLOCK_STEP) * 2) / BLOCK_STEP, 10)
	const height_num = parseInt((shape[1] - (BLOCK_SIZE - BLOCK_STEP) * 2) / BLOCK_STEP, 10)

		for (let k = 0; k < width_num; k++) {

			for (let j = 0; j < height_num; j++) {
				
				let x = k * BLOCK_STEP;
				let y = j * BLOCK_STEP;

				let hr_patch = new cv.Mat(hr_img_array, cv.CV_32FC1);
				let lr_patch = new cv.Mat(lr_img_array, cv.CV_32FC1);

					hr_patch = hr_patch.getRegion(new cv.Rect(y, x, BLOCK_SIZE, BLOCK_SIZE));
					lr_patch = lr_patch.getRegion(new cv.Rect(y, x, BLOCK_SIZE, BLOCK_SIZE));
			
					hr_patch = hr_patch.div(255.0).getDataAsArray()
					lr_patch = lr_patch.div(255.0).getDataAsArray()
				

				let test_lr = tf.tensor2d(hr_patch);

				let convolutial_center = create_conv_center(conv_side, hr_patch);
				
				let lr = tf.zeros([1, Patch_size, Patch_size],'float32');
				let hr = tf.zeros([1, Patch_size, Patch_size],'float32');
				
					data.push([lr_patch]);
					label.push([convolutial_center]);

			}

		}
		console.log(iterator," - image loaded to crop and prepare training set function");
	}

	fs.writeFileSync('../dataset/croped_Data.json', JSON.stringify(data), 'utf8');
	fs.writeFileSync('../dataset/croped_Label.json', JSON.stringify(label), 'utf8');
	return [data, label];
};