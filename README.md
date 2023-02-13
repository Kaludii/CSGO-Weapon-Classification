
# Sentiment Analysis of Customer Reviews

A tool that analyzes the overall sentiment of customer reviews for a specific product or service, whether it's positive or negative. This analysis is performed by using natural language processing algorithms and machine learning from the model `Reviews-Sentiment-Analysis` trained by Kaludi, allowing businesses to gain valuable insights into customer satisfaction and improve their products and services accordingly.

This tool is built using the Gradio library and utilizes the `transformers` library for its machine learning capabilities.

## Model

The sentiment analysis tool uses a pre-trained model 'Reviews-Sentiment-Analysis' available on HuggingFace at [https://huggingface.co/Kaludi/Reviews-Sentiment-Analysis](https://huggingface.co/Kaludi/Reviews-Sentiment-Analysis).

## Dataset

The 'Reviews-Sentiment-Analysis' model was trained on a dataset of customer reviews also available on HuggingFace at [https://huggingface.co/datasets/Kaludi/data-reviews-sentiment-analysis](https://huggingface.co/datasets/Kaludi/data-reviews-sentiment-analysis).

## How to Use

1.  Clone or download the repository.
2.  Install the required libraries by running `pip install -r requirements.txt`.
3.  Run the script using `python app.py`.
4.  Input a customer review in the textbox and click on "Run".
5.  The output will show the sentiment prediction of the review as either Positive or Negative along with the respective confidence score.

## Libraries Used

-   Gradio
-   Transformers
-   Numpy
-   Pandas
-   Pickle
-   Scipy

## Model

The model `Reviews-Sentiment-Analysis` was trained by Kaludi and is available on [HuggingFace](https://huggingface.co/Kaludi).

## Contributor

-   [Kaludi](https://github.com/Kaludii)