const tf = require('@tensorflow/tfjs-node');
// const seedrandom = require('seedrandom');
// const TensorContainer = require('@tensorflow/tfjs-core/dist/tensor_types').TensorContainer;

// const seed = (s) => {return seedrandom(s)};
// const seed = seedrandom.prng;
// console.log(seed);
// process.exit(0);

tf.loadLayersModel("https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json").then((baseModel) => {
    console.log("Model loaded");
    console.log(baseModel.summary());
    // baseModel.compile({optimizer: 'adam', loss: 'categoricalCrossentropy', optimizer: 'sgd'});
    const model = tf.sequential();
    model.add(baseModel);
    const varianceScaling = tf.initializers.varianceScaling({});
    model.add(tf.layers.dense({
        inputShape: [inputSize],
        units: params.denseUnits,
        activation: 'relu',
        kernelInitializer: varianceScaling, // 'varianceScaling'
        useBias: true
    }));
    model.add(tf.layers.dense({
        kernelInitializer: varianceScaling, // 'varianceScaling'
        useBias: false,
        activation: 'softmax',
        units: 7
    }));
    const optimizer = tf.train.adam(0.001);
        // const optimizer = tf.train.rmsprop(params.learningRate);

    model.compile({
        optimizer,
        // loss: 'binaryCrossentropy',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // const trainData = tf.data.Dataset<TensorContainer>();
    // model.add();
    // model.fit({x: x, y: y, epochs: 1});
});
// console.log(model);
// model.then((res) => {
//     console.log("Model loaded");
//     console.log(res.summary());
// })

// const baseModel = () => {tf.loadLayersModel("https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json")};

console.log(baseModel.summary());
