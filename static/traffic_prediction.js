document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('traffic-form');
    const predictionResult = document.getElementById('prediction-result');

    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        const day = parseFloat(document.getElementById('day').value);
        const zone = parseFloat(document.getElementById('zone').value);
        const weather = parseFloat(document.getElementById('weather').value);
        const temperature = parseFloat(document.getElementById('temperature').value);

        // Load your pre-trained machine learning model (replace "your_model_url_here")
        const model = await tf.loadLayersModel("your_model_url_here");

        // Normalize input data (you may need to adjust the normalization process)
        const inputTensor = tf.tensor([[day, zone, weather, temperature]]);
        const normalizedInput = inputTensor.div(tf.scalar(255)); // Normalize to [0, 1]

        // Make a prediction
        const prediction = model.predict(normalizedInput);

        // Inverse normalize the prediction (if necessary)
        const inversePrediction = prediction.mul(tf.scalar(255));

        // Display the prediction result
        predictionResult.textContent = `Predicted Traffic: ${inversePrediction.dataSync()[0].toFixed(2)}`;
    });
});
