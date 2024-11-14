import os
import csv
import datetime as dt
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Add to rquiremnts
# pip install urllib3==1.26.6

# 1. Lade das vortrainierte deutsche BERT-Modell und den Tokenizer
model_name = "oliverguhr/german-sentiment-bert"  # Ein feingetuntes Sentiment-Modell für deutsche Sprache
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 2. Funktion zum Laden eines Datensatzes
def lade_saetze(dateiname):
    """
    Mithilfe der pandas Funktion read_csv wird das csv-File in ein pandas Dataframe
    geladen und die Spalte mit den enthaltenen Feedback-Texten in das List-Objekt
    saetze geschrieben. Das List-Objekt wird an die Funktion zurückgegeben.
    """
    arbeitsverzeichnis= Path.cwd()
    daten_verzeichnis = str(arbeitsverzeichnis.parent) + r"\input_data"
    # dateiname = 'feedback.csv'
    file_to_open = os.path.join(daten_verzeichnis, dateiname)
    df = pd.read_csv(file_to_open, delimiter=";")
    saetze = df['Feedback_Text'].tolist()
    return saetze

# 3. Funktion zur Sentiment-Analyse
def analyse_sentiment(text):
    # Text tokenisieren

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Vorhersage mit dem Modell
    with torch.no_grad():
        outputs = model(**inputs)

    # Die Logits werden in Wahrscheinlichkeiten umgewandelt (softmax)
    probs = F.softmax(outputs.logits, dim=-1)

    # Die Klasse mit der höchsten Wahrscheinlichkeit bestimmen
    sentiment = torch.argmax(probs).item()
    sentiment_label = ["positive", "negative", "neutral"][sentiment]

    return sentiment_label, probs[0][sentiment].item()


# 4. Teste die Funktion mit einem deutschen Satz

saetze =  lade_saetze("feedback.csv")
for satz in saetze:
    #text = "Ich musste stundenlang warten,  bis sich jemand bei mir gemeldet hat. Sehr frustrierend."
    sentiment, confidence = analyse_sentiment(satz)
    print(f"Text: {satz}")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")

