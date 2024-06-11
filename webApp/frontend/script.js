function submitForm() {
    document.getElementById("jsonOutput").style.display = "block";
    document.getElementById("pred").style.display = "block";

    // Clear existing img elements
    const jsonOutput = document.getElementById('jsonOutput');
    while (jsonOutput.firstChild) {
        jsonOutput.removeChild(jsonOutput.firstChild);
    }

    const form = document.getElementById('yesNoForm');
    const formData = new FormData(form);
    const gameData = [];

    for (let i = 1; i <= 31; i++) {
        const isChecked = formData.get(`q${i}`) ? 1 : 0;
        gameData.push(isChecked);
    }

    const payload = { game_data: gameData };

    console.log("Request payload:", payload);

    fetch('http://127.0.0.1:5001/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            const img = document.createElement('img');
            img.setAttribute('alt', data.rating);

            switch (data.rating) {
                case 'Everyone':
                    img.setAttribute('src', 'https://www.esrb.org/wp-content/uploads/2019/05/E.svg');
                    img.setAttribute('alt', "Everyone");
                    break;
                case 'Everyone 10+':
                    img.setAttribute('src', 'https://www.esrb.org/wp-content/uploads/2019/05/E10plus.svg');
                    img.setAttribute('alt', "Everyone 10+");
                    break;
                case 'Teen':
                    img.setAttribute('src', 'https://www.esrb.org/wp-content/uploads/2019/05/T.svg');
                    img.setAttribute('alt', "Teen");
                    break;
                case 'Mature 17+':
                    img.setAttribute('src', 'https://www.esrb.org/wp-content/uploads/2019/05/M.svg');
                    img.setAttribute('alt', "Mature 17+");
                    break;
                default:
                    img.setAttribute('src', ''); // Set a default image or leave it blank
            }

            jsonOutput.appendChild(img);
        })
        .catch(error => {
            console.error('Error:', error);
            // Print the error message or handle it in any way you want
            document.getElementById('jsonOutput').textContent = 'An error occurred. Please try again later.';
        });
}
