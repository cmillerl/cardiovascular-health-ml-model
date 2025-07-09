addEventListener('DOMContentLoaded', function () {

    console.log("Script loaded");
    const maxRetries = 10;
    let prediction = null;
    let form = null;
    let currentRetries = 0;


    function getForm() {
        form = document.getElementById('prediction-form');
        prediction = document.getElementById('prediction-result');

        if (form && prediction) {
            console.log("Form found:", form);
            console.log("Prediction found:", prediction);

            form.addEventListener('submit', async function (e) {
                e.preventDefault();
                console.log("Form submitted");

                const formData = new FormData(form);
                const data = new URLSearchParams(formData);

                try {
                    const response = await fetch("http://localhost:5000/predict", {
                        method: 'POST',
                        body: data,
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded'
                        }
                    });

                    const result = await response.json();
                    prediction.textContent = `Prediction: ${result.prediction}`;
                    console.log("Prediction result:", result);
                } catch (error) {
                    console.error("Error during prediction:", error);
                }
            });
        } else if ((!form || !prediction) && currentRetries < maxRetries) {
            currentRetries++;
            setTimeout(getForm, 100);
            console.log("Form or prediction not found, retrying...");
        }
    }


    getForm();

});