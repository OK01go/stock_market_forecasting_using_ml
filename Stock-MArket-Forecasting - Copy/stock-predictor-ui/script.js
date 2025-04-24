document.getElementById('predictForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const submitButton = this.querySelector('input[type="submit"]');
    submitButton.disabled = true;
    submitButton.value = 'Predicting...';

    const dataField = document.getElementById('data');
    const rawData = dataField.value;
    const prices = rawData.split(',').map(p => parseFloat(p.trim()));

    // Validate exactly 60 numbers
    if (prices.length !== 60 || prices.some(isNaN)) {
        alert("Please enter exactly 60 valid comma-separated numbers.");
        submitButton.disabled = false;
        submitButton.value = 'Predict';
        return;
    }

    const formData = new FormData();
    formData.append('data', prices.join(','));

    // Get selected model
    const selectedModel = document.getElementById('model').value;
    formData.append('model', selectedModel);

    fetch('https://stock-predictor-api.onrender.com/predict', {
    method: 'POST',
    body: formData
})

    .then(response => response.json())
    .then(data => {
        submitButton.disabled = false;
        submitButton.value = 'Predict';

        if (data.error) {
            alert(data.error);
            return;
        }

        const predictions = data.predictions;
        const labels = Array.from({ length: predictions.length }, (_, i) => `Day ${i + 1}`);

        const ctx = document.getElementById('predictionChart').getContext('2d');
        if (window.chart) {
            window.chart.destroy();
        }

        window.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: `Predicted Price (${selectedModel})`,
                    data: predictions,
                    fill: false,
                    borderColor: 'blue',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while fetching predictions.');
        submitButton.disabled = false;
        submitButton.value = 'Predict';
    });
});
