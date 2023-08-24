import argparse
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score

# Function to perform sentiment analysis
def predict_sentiment(text, model, tokenizer):
    # Tokenize and preprocess the input text
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512, pad_to_max_length=True)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension

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
    parser.add_argument("--text", type=str, help="Input text for sentiment analysis")
    parser.add_argument("--test_file", type=str, default=None, help="Path to the test data file (default: None)")
    parser.add_argument("--model_dir", type=str, default="distillbert_sentiment_model", help="Directory containing the trained model (default: 'distillbert_sentiment_model')")
    args = parser.parse_args()

    # Load the trained model and tokenizer
    model_dir = args.model_dir
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)

    if args.test_file:
        # Test on a test data file
        test_data = pd.read_excel(args.test_file)
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
        print(f"Classification report saved to 'result_test_data.txt'")

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
