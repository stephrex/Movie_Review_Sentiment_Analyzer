const analyzeText = () => {
    const review = document.getElementById('movie_review').value;
    const predicted_sentiment = document.getElementById('predicted_sentiment')

    console.log(review)
    axios.post('https://movie-review-sentiment-analyzer-y20q.onrender.com/predict',
        { content: review },
        {
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            console.log(response)
            predicted_sentiment.textContent = response.data.prediction
        })
        .catch(error => {
            console.error("Error fetching prediction:", error);
        });
}
