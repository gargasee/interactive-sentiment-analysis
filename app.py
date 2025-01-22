from flask import Flask, render_template, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.graph_objs as go
import plotly.io as pio

nltk.download('vader_lexicon')

# Initialize Flask app
app = Flask(__name__)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Perform sentiment analysis
    scores = sia.polarity_scores(text)
    sentiment = max(scores, key=scores.get)

    # Create a bar chart using Plotly
    bar_chart = go.Figure(
        [go.Bar(x=list(scores.keys()), y=list(scores.values()), marker=dict(color=['red', 'orange', 'blue', 'green']))]
    )
    bar_chart.update_layout(
        title='Sentiment Analysis', 
        xaxis_title='Sentiment', 
        yaxis_title='Score',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white')
    )

    # Convert the Plotly figure to JSON for rendering on the front end
    graph_json = pio.to_json(bar_chart)

    return jsonify({'sentiment': sentiment, 'graph': graph_json})

if __name__ == '__main__':
    app.run(debug=True)