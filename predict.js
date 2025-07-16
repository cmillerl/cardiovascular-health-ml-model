/**
 * This script is responsible for handling the prediction functionality of the cardiovascular disease model.
 *
 * It retrieves the form input from the user, sends it to the back-end server for processing, and displays
 * the prediction result on the web page to the user without reloading the webpage.
 **/
addEventListener('DOMContentLoaded', function () {

    console.log("Predict script loaded");
    const maxRetries = 10; // Maximum number of retries.
    let prediction = null; // Element that displays the prediction result.
    let form = null; // Form element.
    let currentRetries = 0; // Current number of retries.

    /**
     * This function attempts to retrieve the form and prediction elements from the DOM.

     * If found, attach an event listener to the form to handle the prediction request,
     * preventing the default form submission behavior.

     * A POST request is sent to the back-end server with the form data, and the prediction 
     * result is returned and displayed to the user inside the prediction element.

     * If either element is not found, it retries after a short delay until the maximum number of
     * retries is reached or the elements are found.
     **/
    function getForm() {
        form = document.getElementById('prediction-form');
        prediction = document.getElementById('prediction-result');

        // If both elements exist, output a message to the console and continue.
        if (form && prediction) {
            console.log("Form found:", form);
            console.log("Prediction found:", prediction);

            // Add an event listener (submit) to the form to handle the prediction request.
            // This will prevent the default form submission behavior (e.preventDefault()).
            form.addEventListener('submit', async function (e) {
                e.preventDefault();
                console.log("Form submitted");

                // Collects the form data and stores it in a formData object.
                const formData = new FormData(form);
                // Converts the formData to a URLSearchParams object for better handling in the fetch request.
                const data = new URLSearchParams(formData);

                try {
                    // Sends a POST request to the Flask API endpoint with the form data.
                    const response = await fetch("http://localhost:5000/predict", {
                        method: 'POST',
                        body: data,
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded'
                        }
                    });

                    // Retrieve the response from the server.
                    const result = await response.json();
                    // Displays the prediction result in the prediction element.
                    prediction.textContent = `Prediction: ${result.prediction}`;
                    console.log("Prediction result:", result);
                } catch (error) {
                    // If an error occurs during the fetch request, log the error to the console.
                    console.error("Error during prediction:", error);
                }
            });
            // If the form and prediction elements are not found or the maximum number of retries has been reached, stop retrying.
        } else if ((!form || !prediction) && currentRetries < maxRetries) {
            currentRetries++;
            setTimeout(getForm, 100);
            console.log("Form or prediction not found, retrying...");
        }
    }

    // Start the process of getting the form and prediction elements.
    getForm();

});