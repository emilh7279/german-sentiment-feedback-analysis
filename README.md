# 📊 German Sentiment Analysis for Service Desk Feedback with `TextBlob-de`, Hugging Face's BERT, and Custom Self-Trained Models

Hi everyone,

I wanted to let you know that I had to set the old Git repository (from August 2024) to private for specific reasons related to data integrity and security. I apologize for any inconvenience this change may have caused.

To ensure you still have access to everything, I've created a new repository. You can find it at the following link:

[https://github.com/emilh7279/german-sentiment-feedback-analysis]

Please feel free to reach out if you encounter any issues accessing the new repository or if there’s anything else I can help with. Thank you for your understanding!

Best regards,
Emil

Welcome to the **German Sentiment Analysis** project, where we utilize **`TextBlob-de`**, **Hugging Face's BERT**, and **Custom Finetuned Models** for analyzing customer feedback from a service desk. The goal is to detect and classify sentiments (positive, negative, neutral) in feedback and use the results to generate Key Performance Indicators (KPIs) for tracking service desk performance and customer satisfaction.


----------

## 🎯 Project Background

Customer feedback from a service desk is invaluable for improving services. Whether it's praise for excellent support or dissatisfaction with unresolved issues, understanding customer sentiment allows for more informed decisions. This project focuses on analyzing German-language feedback from service desk interactions using **rule-based**, **pre-trained**, and **custom-trained deep learning models**.

The main goal of this project is to perform sentiment analysis on German-language customer feedback and create KPIs to measure:

-   The ratio of positive to negative feedback over time.
-   The percentage of neutral feedback, indicating customer indifference or satisfaction.
-   Sentiment trends that can help monitor service desk agent performance and customer satisfaction across different departments or time periods.

These KPIs will enable businesses to track the effectiveness of their customer support services, identify pain points, and optimize resource allocation.


----------

## 🚀 Features

-   **TextBlob-de**: A rule-based sentiment analysis tool for the German language.
-   **BERT** from Hugging Face: A pre-trained transformer model fine-tuned for sentiment analysis.
-   **Custom Finetuned Models**: Models trained on company-specific service desk feedback for better domain adaptation.

----------

## 📚 Table of Contents

1.  [Installation](#installation)
2.  [Usage](#usage)
    -   [TextBlob-de Sentiment Analysis](#textblob-de-sentiment-analysis)
    -   [BERT Sentiment Analysis](#bert-sentiment-analysis)
    -   [Custom Finetuned Model Analysis](#custom-finetuned-bert-model-analysis)
3.  [License](https://github.com/emilh7279/german-sentiment-feedback-analysis/blob/master/LICENSE)

----------

## 🛠 Installation

### 1. Clone the repository:

    git clone https://github.com/emilh7279/german-feedback-analysis.git
cd german-sentiment-analysis

### 2. Create a virtual environment:

	  python3 -m venv venv
    source venv/bin/activate  # For Windows: venv\Scripts\activate


### 3. Install the dependencies:

    pip install -r requirements.txt` 

The following packages will be installed:

-   `textblob-de`
-   `transformers`
-   `torch`
-   `pandas`
-   `scikit-learn`

----------

## 💡 Usage

This repository provides three approaches to performing sentiment analysis on service desk feedback data: **TextBlob-de**, **BERT**, and **Finetuned BERT Models**. The idea is to use the results of the sentiment analysis to generate KPIs to track customer satisfaction.

### TextBlob-de Sentiment Analysis

`TextBlob-de` is a rule-based model that assigns polarity values to words and computes the overall sentiment of the text.

#### Example Usage: Feedback from a service desk

    from textblob_de import TextBlobDE
    
    feedback = "Der Service war schlecht, ich habe nie eine Antwort erhalten."
    
    blob = TextBlobDE(feedback)
    print("Sentiment Polarity:", blob.sentiment.polarity)` 

**Output:**
`Sentiment Polarity: -0.6`

### Folder textblob-de
The folder textblob-de contains two python files which shows the usage  of a sentiment analysis.

#### File simple_textblob.py
 ##### Function: `satz_bewertung(satz)`

This function evaluates the sentiment of a given German sentence and returns an assessment as **positive**, **negative**, or **neutral**. The sentiment is determined based on the **polarity** of the sentence, which is calculated through sentiment analysis.

##### Parameters:

-   `satz`: A string representing the German sentence to be analyzed.

#### How it works:

1.  **Create a TextBlob object**: First, a `TextBlob` object is created from the input sentence. This object allows the use of sentiment analysis methods.
    
2.  **Analyze polarity**: The function analyzes the polarity of the sentence using the method `blob.sentiment.polarity`. The polarity is a value ranging from -1 to 1:
    
    -   Values greater than 0 indicate that the sentence is positive.
    -   Values less than 0 indicate that the sentence is negative.
    -   A value of exactly 0 indicates that the sentence is neutral.
3.  **Sentiment evaluation**:
    
    -   If the polarity is positive (i.e., greater than 0), the function returns that the sentence is positive, along with the exact polarity value.
    -   If the polarity is negative (i.e., less than 0), the sentence is classified as negative.
    -   If the polarity is exactly 0, the sentence is classified as neutral.

#### Return:

-   A string that describes the sentiment of the sentence ("positive", "negative", "neutral") and includes the calculated polarity value.

#### Sample Input:

 1. Der Service war hervorragend und mein Anliegen wurde sofort geklärt.
 2. Der Prozess war etwas langwierig, aber das Endergebnis war in Ordnung.  
 3. Der Mitarbeiter war sehr unhöflich und mein Problem wurde nicht 	   behoben.
 4. Schlechter Kundenservice, niemand hat auf meine Anfragen geantwortet.
 5. Der Service hat zu lange gebraucht und mein Problem nicht gelöst.
 6. Ich bin wirklich beeindruckt, wie schnell mein Problem gelöst wurde.
 7. Fantastischer Service! Mein Problem wurde sofort gelöst.
 8. Es hat viel zu lange gedauert und niemand konnte mir helfen.
 9. Es war okay, nicht das Beste, aber das Problem wurde gelöst.
 10. Das Support-Team schien verwirrt und hat nicht wirklich geholfen.

#### Sample Output:

 1. Der Satz ist positiv. Polarität: 1.0
 2. Der Satz ist neutral. Polarität: 0.0
 3. Der Satz ist negativ. Polarität: -1.0
 4. Der Satz ist negativ. Polarität: -1.0
 5. Der Satz ist neutral. Polarität: 0.0
 6. Der Satz ist positiv. Polarität: 0.85
 7. Der Satz ist positiv. Polarität: 0.5
 8. Der Satz ist positiv. Polarität: 0.35
 9. Der Satz ist neutral. Polarität: 0.0
 10. Der Satz ist negativ. Polarität: -0.75

--------

### File sentiment_feedback_textblob.py
This script includes, in addition to the satz_bewertung(satz) function,
functions for reading a larger dataset in CSV format and saving the determined sentiments in a CSV file.

#### Function: lade_saetze(dateiname)
This function loads sentences from a CSV file that contains feedback data. The filename is passed as a parameter,
and the function extracts the text column from the CSV file and returns it as a list of sentences.

> [!IMPORTANT]  
> The CSV file must be placed in the input_data folder.

#### Parameters:
dateiname: A string that specifies the name of the CSV file containing the feedback data.

#### Function: speichere_ergebnisse(ergebnisse)
This function saves the results of an analysis into a CSV file.
The results are stored in a structured way, making them easy to view or process later.

> [!NOTE]
> The result CSV file is located in textblob-de/results.

--------

#### Conclusion TextBlob-DE
As you can see TextBlob is a fast and easy-to-use method for performing sentiment analysis on sentences.
With textblob-de, the framework also provides a direct way to interpret German sentences.

--------

## BERT Sentiment Analysis

**BERT** (Bidirectional Encoder Representations from Transformers) is a powerful deep learning model developed
by Google for Natural Language Processing (NLP). It revolutionized the NLP field by its ability to understand
the context of words in a sentence from both the left and right sides. This bidirectional analysis allows BERT
to capture complex meanings and nuances in texts more effectively than previous models.

-   **Pre-training**:
    
    -   BERT is trained on a large corpus of unlabeled text to learn language patterns and structures.

-   **Fine-tuning**:
    
    -   After pre-training, BERT can be adapted for specific tasks, such as sentiment analysis.
    This is done by training the model on a labeled dataset that includes examples of positive, negative, and neutral
    sentiments. Through fine-tuning, the model learns to account for specific features of the target task.

### Steps for Performing Sentiment Analysis with BERT

1.  **Loading the Pre-trained BERT Model**:
    -   We use the Hugging Face pipeline to load a BERT model trained for sentiment analysis. An example model for the German language is `oliverguhr/german-sentiment-bert`.
2.  **Preprocessing Input Data**:
    -   Input data (customer feedback) needs to be preprocessed. This includes tokenization and adding special tokens required for the model. The text is transformed into a format that can be processed by the model.
3.  **Predicting the Sentiment Class**:
    -   After preprocessing, the text is fed into the model, which outputs a prediction for the sentiment of the text. The model returns a list of sentiment classes along with the probabilities for each class.
4.  **Interpreting Results**:
    -   The predictions are analyzed to evaluate the overall sentiment of customer feedback. The obtained values can provide insights into customer satisfaction.

### Code and Examples

The Python file hugging_face_entiment/simple_sentiment_hugging_face_bert.py contains a simple implementation of BERT and loads a pre-trained model for sentiment analysis.

#### 1. Loading the Pre-trained Model and Tokenizer

    model_name = "oliverguhr/german-sentiment-bert"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name) 

-   Here, the pre-trained German sentiment BERT model (`oliverguhr/german-sentiment-bert`) is loaded.
-   **AutoTokenizer**: Splits the text into tokens that the model understands.
-   **AutoModelForSequenceClassification**: Loads the sentiment model for text classification.

#### 2. Loading the Dataset

    def lade_saetze(dateiname):
        arbeitsverzeichnis = Path.cwd()
        daten_verzeichnis = str(arbeitsverzeichnis.parent) + r"\input_data\feedback"
        file_to_open = os.path.join(daten_verzeichnis, dateiname)
        df = pd.read_csv(file_to_open, delimiter=";")
        saetze = df['Feedback_Text'].tolist()
        return saetze

-   This function loads feedback text from a CSV file using **pandas**.
-   The CSV file is read, and the `Feedback_Text` column is converted into a list and returned.

#### 3. Sentiment Analysis Function

    def analyse_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = F.softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(probs).item()
        sentiment_label = ["positive", "negative", "neutral"][sentiment]
        
        return sentiment_label, probs[0][sentiment].item()

-   **analyse_sentiment(text)** performs sentiment analysis on the given text.
-   The text is first tokenized using the **tokenizer**.
    -   **return_tensors="pt"** returns the tokens as PyTorch tensors.
    -   **truncation** and **padding** ensure the input data has the correct length.
-   The model predicts the sentiment of the text without calculating gradients (using **torch.no_grad()** to save computational resources).
-   The model output (logits) is converted into probabilities using the **softmax** function.
-   The label with the highest probability is determined (using **torch.argmax**) and translated into `sentiment_label` as "positive", "negative", or "neutral".
-   The function returns both the predicted sentiment and the probability for that prediction.

#### 4. Testing the Code with Example Sentences

    saetze =  lade_saetze("feedback.csv")
    for satz in saetze:
        sentiment, confidence = analyse_sentiment(satz)
        print(f"Text: {satz}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")

#### Sample Output:
1. Der Service war hervorragend und mein Anliegen wurde sofort geklärt.
Sentiment: positive (Confidence: 0.9997)
2. Der Prozess war etwas langwierig, aber das Endergebnis war in Ordnung.
Sentiment: neutral (Confidence: 0.9934)
3. Der Mitarbeiter war sehr unhöflich und mein Problem wurde nicht behoben.
Sentiment: negative (Confidence: 0.9990)
4. Schlechter Kundenservice, niemand hat auf meine Anfragen geantwortet.
Sentiment: negative (Confidence: 0.9982)
5. Der Service hat zu lange gebraucht und mein Problem nicht gelöst.
Sentiment: negative (Confidence: 0.9857)
6. Ich bin wirklich beeindruckt, wie schnell mein Problem gelöst wurde.
Sentiment: positive (Confidence: 0.9974)
7. Fantastischer Service! Mein Problem wurde sofort gelöst.
Sentiment: positive (Confidence: 0.9970)
8. Es hat zu lange gedauert und niemand konnte mir helfen.
Sentiment: negative (Confidence: 0.9930)
9. Es war okay, nicht das Beste, aber das Problem wurde gelöst.
Sentiment: positive (Confidence: 0.9615)
10. Das Support-Team schien verwirrt und hat nicht wirklich geholfen.
Sentiment: negative (Confidence: 0.9912)

## Custom Finetuned BERT Model Analysis

Fine-tuning a BERT model allows us to adapt the general-purpose, pre-trained BERT architecture to a specific task or dataset.
While BERT is pre-trained on massive amounts of general language data, fine-tuning helps it perform better on specific tasks—like sentiment analysis for German customer feedback in this project.
During fine-tuning, we further train the model on our labeled data, enabling it to capture domain-specific language patterns, vocabulary, and nuances more effectively.

In this project, we use a pre-trained German BERT model from the Hugging Face library and fine-tune it for sentiment classification, categorizing text into positive, negative, or neutral sentiment.
By tailoring BERT to our dataset, we aim to increase model accuracy and better understand customer sentiment trends.

### Code and Examples
We split the task of applying a custom fine-tuned BERT model into two scripts. In the first script, the custom fine-tuned BERT model is trained and saved.
In the second script, we load the custom fine-tuned model and use it to evaluate our new datasets.

### Generate Custome Finetuned BERT Model (finetune_model.py)

#### 2. Load labeled CSV-Dataset

#### 3. Load Tokenizer

#### 4. Tokenize the Dataset

    def tokenize_function(examples):
        return tokenizer(examples['Text'], padding="max_length", truncation=True)               
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

-   **Tokenization function**: A function that converts text into tokens. The text should be in a column named `"Text"`.
-   **Tokenize entire dataset**: `dataset.map()` applies the tokenization function to all data, with `batched=True` allowing for batch processing.

#### 5. Split into Training and Test Data

    train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

-   **Splitting the dataset**: The dataset is split into 80% training and 20% test data.

#### 6. Load the Model

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

-   **Load model**: A pre-trained model for sentiment classification is loaded. Here it’s assumed there are three classes (e.g., positive, negative, and neutral sentiments), so `num_labels=3` is set.

#### 7. Configure Training Parameters

    training_args = TrainingArguments(
        output_dir="finetuned_hugging_face_sentiment/results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

-   **`TrainingArguments`**: These parameters control the model training:
    -   **`output_dir`**: Folder to save results.
    -   **`evaluation_strategy`**: Evaluates the model at the end of each epoch.
    -   **`learning_rate`**: Learning rate, which controls how fast the model learns.
    -   **`per_device_train_batch_size` and `per_device_eval_batch_size`**: Batch sizes for training and evaluation.
    -   **`num_train_epochs`**: Number of training epochs.
    -   **`weight_decay`**: Regularization to help avoid overfitting.

#### 8. Initialize the Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

-   **Trainer**: Initializes the training by passing the model, training arguments, and datasets.

#### 9. Train the Model

    trainer.train()

-   **Training**: Starts training the model.


#### 11. Save the Fine-Tuned Model

    model.save_pretrained("finetuned_hugging_face_sentiment/finetuned_models")
    tokenizer.save_pretrained("finetuned_hugging_face_sentiment/finetuned_models")

### Use Custome Finetuned BERT Model (use_finetuned_hugging_face_bert.py)

This Python script processes text data from a CSV file to perform sentiment analysis using a pre-trained transformer model.
This script is designed for batch sentiment analysis, making it useful for applying a machine learning model to a large number
of text entries in a CSV file and saving the results in an output file

Here’s a breakdown of each function:

#### 1. load_csv(file_path)

- Loads a CSV file from the specified file_path and returns a DataFrame (df).


#### 2. apply_model_to_texts(texts, model, tokenizer)
    
This function applies a sentiment analysis model to a list of texts. Here’s how it works:
- Initialize Predictions List: An empty list, predictions, is created to store the results.
- Tokenize Each Text: For each text, the tokenizer converts it into tokens, preparing it for model input. It uses padding and truncation to keep the input length within a specified limit (max_length=128).
- Run the Model: The model processes the tokenized input without updating gradients (torch.no_grad()), which conserves memory and speeds up inference.
- Compute Probabilities: The model returns logits (raw prediction values) for each class. softmax is applied to convert these logits into probabilities.
- Determine Prediction: The torch.argmax function finds the class with the highest probability, which is the predicted sentiment. The confidence score is extracted as the probability of this predicted class.
- Map Predicted Class to Label: Maps the predicted class (0, 1, 2) to a sentiment label ("negative", "neutral", "positive").
- Store Predictions: Appends each prediction as a tuple of (predicted_label, confidence) to the predictions list.
- Returns a list of predictions for all texts.

#### 3. save_predictions_to_csv(df, predictions, output_file)

Adds prediction results to the original DataFrame df as new columns.
- Sentiment: Contains the sentiment label (e.g., "negative", "neutral", "positive").
- Confidence: Contains the confidence score for each prediction.
- Saves this updated DataFrame to a new CSV file specified by output_file.

#### 4. apply_model_to_csv(input_csv, output_csv, model_name)

Main function to apply the sentiment model to a CSV file.
- Load Model and Tokenizer: Loads a fine-tuned transformer model and its tokenizer from model_name.
- Set Model to Evaluation Mode: Configures the model in evaluation mode (disables dropout layers).
- Load CSV: Uses load_csv to load data from input_csv.
- Verify Column: Ensures the DataFrame has a "Text" column; raises an error if missing.
- Apply Model to Texts: Calls apply_model_to_texts to generate predictions on the "Text" column.
- Save Predictions: Uses save_predictions_to_csv to save the prediction results to output_csv.
