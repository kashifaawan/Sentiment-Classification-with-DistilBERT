# Sentiment-Classification-with-DistilBERT

## Introduction

This repository contains code for training a sentiment analysis model using the DistilBERT architecture. The model is designed to classify text as either positive or negative sentiment. 
Detail instructions are provided on how to train your own sentiment analysis model using this code.

## Methodology

### Data Collection and Preprocessing

The training dataset used in this project consists of labeled text data, where each sample is assigned a sentiment label (positive or negative). The dataset was obtained from [source link](https://github.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k). It includes two Excel files, 'train.xlsx' and 'test.xlsx', each containing 25,000 movie reviews and their corresponding sentiment labels.

Before feeding the data to the model for training, several data preprocessing steps were executed, including text tokenization, padding, and label mapping. This preprocessing ensures that the data is in the appropriate format for training the sentiment analysis model.

### Model Architecture

Model was fine-tuned by adding a classification head for binary sentiment classification (positive or negative). During training, the AdamW optimizer and cross-entropy loss as the objective function was utilized.

### Training and Evaluation

The model was trained for a fixed number of epochs, and its performance was regularly evaluated on a validation data to monitor its progress. 

## Usage

To train your own sentiment analysis model using this code, follow these steps:

1. Clone this repository to your local machine.

2. Install the required packages by running:
   pip install -r requirements.txt

3. Prepare your training data in an Excel file similar to the provided 'train.xlsx' file. Ensure that it includes 'Reviews' and 'Sentiment' columns.

4. Open a terminal and navigate to the repository directory.

5. Run the training script with the desired options:

python train_sentiment_analysis.py --batch_size <batch_size> --epochs <num_epochs> --train_file <path_to_train_file> --output_dir <output_directory>

Replace `<batch_size>`, `<num_epochs>`, `<path_to_train_file>`, and `<output_directory>` with your preferred values.

6. Monitor the training progress, and the model will be saved in the specified output directory once training is complete.

## Requirements

- Python 3.x
- PyTorch
- Transformers library
- pandas
- scikit-learn
- matplotlib

## Download trained Model

You can download the fine-tuned DistilBERT sentiment analysis model from the following link:

[Download Model](https://drive.google.com/file/d/1R29vyVtVdu0xKbwg1DEF7OhPRIRdKWKY/view?usp=drive_link)

The model can be loaded for inference using the provided `inference_sentiment_analysis.py` script. If you need assistance with loading the model, please refer to the [Inference](#inference-with-inference_sentiment_analysispy) section for usage instructions.

Note: Ensure that extracted model is placed in your working directory, otherwise specify appropriate path for it in `inference_sentiment_analysis.py`.

## Inference with inference_sentiment_analysis.py

To perform sentiment analysis on new text data or evaluate the model on a test dataset, you can use the inference_sentiment_analysis.py script. 
This script allows you to load your trained DistilBERT sentiment analysis model and use it for inference.

# Usage

To use the inference_sentiment_analysis.py script, follow the instructions below:

1. Perform Sentiment Analysis on Input Text:

You can use the script to analyze the sentiment of a single input text. Here's the command:

python inference_sentiment_analysis.py --text "Your input text goes here"

Replace "Your input text goes here" with the text you want to analyze. The script will display the input text and the predicted sentiment.

2. Evaluate the Model on a Test Data File:

If you have a test dataset in the same format as the training data (with a 'Reviews' column), you can evaluate the model's performance on it. Here's the command:

python inference_sentiment_analysis.py --test_file /path/to/test_data.xlsx

Replace /path/to/test_data.xlsx with the path to your test data file. The script will generate a classification report and save it to a file named result_test_data.txt. The report includes precision, recall, F1-score, and accuracy metrics for both negative ('neg') and positive ('pos') sentiments.

Note: Ensure that the test data file has the same column structure as the training data, including the 'Sentiment' column.

Additional Options:

--model_dir (optional): You can specify the directory containing the trained model using the --model_dir option. By default, it assumes the model is located in the 'distillbert_sentiment_model' directory.

---

For more details on the code implementation and model training, please refer to the code files and the 'result_test_data.txt' file.

Feel free to contribute to this repository or adapt the code for your specific NLP tasks.
