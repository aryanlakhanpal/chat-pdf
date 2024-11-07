import torch
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline

# Load BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Check if CUDA is available and move model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summarize_text(text, max_length=150, min_length=30):
    """
    Summarizes the input text using the BART model.
    
    Parameters:
    - text (str): The text to summarize.
    - max_length (int): The maximum length of the summary.
    - min_length (int): The minimum length of the summary.
    
    Returns:
    - str: The summarized text.
    """
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def get_user_input():
    """
    Prompts the user for text input.
    
    Returns:
    - str: The input text from the user.
    """
    print("Enter the text to summarize (or type 'exit' to quit):")
    text = ""
    while True:
        line = input()
        if line.strip().lower() == "exit":
            break
        text += line + " "
    return text.strip()

def summarize_multiple_texts(text_list, max_length=150, min_length=30):
    """
    Summarizes a list of texts using the BART model.
    
    Parameters:
    - text_list (list): A list of strings to summarize.
    - max_length (int): The maximum length of the summary.
    - min_length (int): The minimum length of the summary.
    
    Returns:
    - list: A list of summarized texts.
    """
    summaries = []
    for text in text_list:
        summary = summarize_text(text, max_length=max_length, min_length=min_length)
        summaries.append(summary)
    return summaries

if __name__ == "__main__":
    print("Text Summarization using BART Model")
    print("Choose an option:")
    print("1. Summarize pre-defined text examples")
    print("2. Enter your own text to summarize")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Example texts to summarize
        texts = [
            "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term 'artificial intelligence' is often used to describe machines (or computers) that mimic 'cognitive' functions that humans associate with the human mind, such as 'learning' and 'problem solving'.",
            "Machine learning is a branch of artificial intelligence (AI) focused on building applications that learn from data and improve their accuracy over time without being programmed to do so. In data-driven tasks, it’s usually better to use a machine learning model trained on a large dataset instead of relying on hand-coded rules. Such models can generalize more effectively, handling complex patterns in large datasets.",
            "Natural language processing (NLP) is a field of artificial intelligence in which computers analyze, understand, and derive meaning from human language in a smart and useful way. By utilizing NLP, developers can organize and structure knowledge to perform tasks such as automatic summarization, translation, named entity recognition, relationship extraction, sentiment analysis, speech recognition, and topic segmentation."
        ]
        
        print("Summarizing pre-defined texts:")
        summarized_texts = summarize_multiple_texts(texts)
        for i, summary in enumerate(summarized_texts):
            print(f"\nSummary {i + 1}:\n{summary}\n")

    elif choice == "2":
        # Get user input and summarize
        user_input = get_user_input()
        if user_input:
            print("\nSummarizing your text...")
            summary = summarize_text(user_input)
            print("\nSummary:\n", summary)

    else:
        print("Invalid choice. Please enter 1 or 2.")
