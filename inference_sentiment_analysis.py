import argparse
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score

# Tokenize and preprocess the input text
def tokenize_text(text,tokenizer):
        if isinstance(text, str):
            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512, pad_to_max_length=True)
        elif isinstance(text, (list, tuple)) and all(isinstance(item, str) for item in text):
            tokens = tokenizer.encode(" ".join(text), add_special_tokens=True, truncation=True, max_length=512, pad_to_max_length=True)
        else:
            tokens = []  # Skip tokenization for invalid input
        return tokens

# Function to perform sentiment analysis
def predict_sentiment(text, model, tokenizer):
    
    tokens = tokenize_text(text, tokenizer)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(model.device)  # Move to the model's device
    
    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids)
    
    logits = outputs.logits
    predicted_class = logits.argmax(dim=1).item()

    # Map the numerical label back to text (if needed)
    label_map = {0: 'neg', 1: 'pos'}
    predicted_sentiment = label_map[predicted_class]

    return predicted_sentiment

def main():
    parser = argparse.ArgumentParser(description="Perform sentiment analysis using a DistilBERT model.")
    parser.add_argument("--text", type=str, default=None,help="Input text for sentiment analysis")
    parser.add_argument("--test_file", type=str, default=None, help="Path to the test data file (default: None)")
    parser.add_argument("--model_dir", type=str, default="distillbert_sentiment_model", help="Directory containing the trained model (default: 'distillbert_sentiment_model')")
    args = parser.parse_args()
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for inference: {device}")

    # Load the trained model and tokenize
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir) # Load model
    model=model.to(device) # Move the model to the specified device
    if args.test_file:
        # Load test data file
        test_data = pd.read_excel(args.test_file)
        test_data['tokenized'] = test_data['Reviews'].apply(lambda x: tokenize_text(x, tokenizer))

        # Remove rows with empty tokenized sequences (invalid input)
        test_data = test_data[test_data['tokenized'].apply(len) > 0]
        test_data['Predicted Sentiment'] = test_data['Reviews'].apply(lambda x: predict_sentiment(x, model, tokenizer))

        # Calculate the classification report
        true_labels = test_data['Sentiment']
        predicted_labels = test_data['Predicted Sentiment']
        report_dict = classification_report(true_labels, predicted_labels, target_names=['neg', 'pos'], output_dict=True)

        # Calculate accuracy separately
        accuracy = accuracy_score(true_labels, predicted_labels)

        # Write the classification report to a file
        with open('result_test_data.txt', 'w') as report_file:
            for label, metrics in report_dict.items():
                if label in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                report_file.write(f"Label: {label}\n")
                report_file.write(f"Precision: {metrics['precision']:.2f}\n")
                report_file.write(f"Recall: {metrics['recall']:.2f}\n")
                report_file.write(f"F1-Score: {metrics['f1-score']:.2f}\n\n")

            # Add accuracy to the report
            report_file.write(f"Accuracy: {accuracy:.2f}\n")
        print(f"Classification report saved to 'result_test_data.txt")
    elif args.text:
        # Perform sentiment analysis on input text
        input_text = args.text
        predicted_sentiment = predict_sentiment(input_text, model, tokenizer)
        print(f"Input Text: {input_text}")
        print(f"Predicted Sentiment: {predicted_sentiment}")

    else:
        print("Please provide either --text or --test_file argument.")

if __name__ == "__main__":
    main()
