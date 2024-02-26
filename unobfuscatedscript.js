const URL = 'https://storage.googleapis.com/tm-model/3Qe6xLKFT/';
const URL2 = 'https://storage.googleapis.com/tm-model/in84lxb73/';
const URL3 = 'https://storage.googleapis.com/tm-model/nZDP1oM63/';

let model, labelContainer, maxPredictions;
let model2, labelContainer2, maxPredictions2;

// Load the image model
async function init() {
    const modelURL = URL + 'model.json';
    const metadataURL = URL + 'metadata.json';
    const model2URL = URL2 + 'model.json';
    const metadata2URL = URL2 + 'metadata.json';
    const model3URL = URL3 + 'model.json';
    const metadata3URL = URL3 + 'metadata.json';

    // load the model and metadata
    model = await tmImage.load(modelURL, metadataURL);
    model2 = await tmImage.load(model2URL, metadata2URL);
    
    maxPredictions = model.getTotalClasses();
    maxPredictions2 = model2.getTotalClasses();

    labelContainer = document.getElementById('label-container');
    labelContainer2 = document.getElementById('label-container-2');
    for (let i = 0; i < maxPredictions; i++) {
        // and class labels
        labelContainer.appendChild(document.createElement('div'));
    }
    for (let i = 0; i < maxPredictions2; i++) {
        // and class labels
        labelContainer2.appendChild(document.createElement('div'));
    }
}

async function predict(isRequest) {
    var image = document.getElementById('imagePreview');
    const prediction = await model.predict(image, false);
    const prediction2 = await model2.predict(image, false);
    for (let i = 0; i < maxPredictions; i++) {
        labelContainer.childNodes[i].innerHTML = "";
    }
    for (let i = 0; i < maxPredictions2; i++) {
        labelContainer2.childNodes[i].innerHTML = "";
    }
    for (let i = 0; i < maxPredictions; i++) {
        labelContainer.childNodes[i].style.color = 'black';
        labelContainer.childNodes[i].style.fontWeight = 'normal';
    }
    for (let i = 0; i < maxPredictions2; i++) {
        labelContainer2.childNodes[i].style.color = 'black';
        labelContainer2.childNodes[i].style.fontWeight = 'normal';
    }
    for (let i = 0; i < maxPredictions; i++) {
        const classPrediction = prediction[i].className + ': ' + prediction[i].probability.toFixed(4);
        labelContainer.childNodes[i].innerHTML = classPrediction;
    }
    for (let i = 0; i < maxPredictions2; i++) {
        const classPrediction = prediction2[i].className + ': ' + prediction2[i].probability.toFixed(4);
        labelContainer2.childNodes[i].innerHTML = classPrediction;
    }
    let maxIndex = 0;
    let maxIndex2 = 0;
    for (let i = 0; i < maxPredictions; i++) {
        if (prediction[i].probability.toFixed(4) > prediction[maxIndex].probability.toFixed(4)) {
            maxIndex = i;
        }
    }
    for (let i = 0; i < maxPredictions2; i++) {
        if (prediction2[i].probability.toFixed(4) > prediction2[maxIndex2].probability.toFixed(4)) {
            maxIndex2 = i;
        }
    }
    labelContainer.childNodes[maxIndex].style.color = 'green';
    labelContainer.childNodes[maxIndex].style.fontWeight = 'bold';
    labelContainer2.childNodes[maxIndex2].style.color = 'green';
    labelContainer2.childNodes[maxIndex2].style.fontWeight = 'bold';
    document.getElementById('resultPlace').innerHTML = '<h3 id="result" class="text-center" style="font-weight: bold;"></h3>';
    document.getElementById('resultPlace2').innerHTML = '<h3 id="result2" class="text-center" style="font-weight: bold;"></h3>';
    document.getElementById('result').innerHTML = 'Spectral Class: ' + prediction[maxIndex].className + '<br><h4>Confidence: ' + (prediction[maxIndex].probability.toFixed(4) * 100) + '%</h4><br>';
    let starTypes = ['Supergiant', 'Bright Giant', 'Giant', 'Subgiant', 'Main Sequence']
    document.getElementById('result2').innerHTML = 'Luminosity Class: ' + prediction2[maxIndex2].className + '<br>' + starTypes[maxIndex2] + '<br><h4>Confidence: ' + (prediction2[maxIndex2].probability.toFixed(4) * 100) + '%</h4>';
    document.getElementById("imageURL").value = "";
    return {
        result: prediction[maxIndex].className,
        isRequest: isRequest
    };
}