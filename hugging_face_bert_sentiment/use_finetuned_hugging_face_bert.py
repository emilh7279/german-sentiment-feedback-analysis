import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


# Funktion zum Laden der CSV-Datei
def load_csv(file_path):
    # Lese die CSV-Datei ein (Annahme: die Texte stehen in einer Spalte namens 'Text')
    df = pd.read_csv(file_path)
    return df


# Funktion zum Anwenden des Modells auf den Text
def apply_model_to_texts(texts, model, tokenizer):
    predictions = []

    for text in texts:
        # Tokenisiere den Text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

        # Führe die Vorhersage durch
        with torch.no_grad():
            outputs = model(**inputs)

        # Logits erhalten und Wahrscheinlichkeiten berechnen
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)

        # Die Vorhersageklasse (Index der höchsten Wahrscheinlichkeit) ermitteln
        predicted_class = torch.argmax(probs, dim=-1).item()

        # Confidence der Vorhersage (maximale Wahrscheinlichkeit)
        confidence = probs[0][predicted_class].item()

        # Zuordnung der Vorhersageklassen zu Labels (z.B. 0=negativ, 1=neutral, 2=positiv)
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        predicted_label = label_map[predicted_class]

        predictions.append((predicted_label, confidence))

    return predictions


# Funktion zum Speichern der Ergebnisse in eine CSV-Datei
def save_predictions_to_csv(df, predictions, output_file):
    # Füge die Vorhersagen zur DataFrame hinzu
    df['Sentiment'] = [pred[0] for pred in predictions]
    df['Confidence'] = [pred[1] for pred in predictions]

    # Speichere die DataFrame in eine neue CSV-Datei
    df.to_csv(output_file, index=False)
    print(f"Ergebnisse wurden in {output_file} gespeichert.")


# Hauptfunktion zum Anwenden des Modells auf eine CSV-Datei
def apply_model_to_csv(input_csv, output_csv, model_name):
    # Lade das finetunte Modell und den Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Setze das Modell in den Evaluationsmodus
    model.eval()

    # Lade die CSV-Datei
    df = load_csv(input_csv)

    # Prüfe, ob die Spalte 'Text' existiert
    if 'Text' not in df.columns:
        raise ValueError("Die CSV-Datei muss eine Spalte mit dem Namen 'Text' enthalten.")

    # Wende das Modell auf die Texte an
    texts = df['Text'].tolist()
    predictions = apply_model_to_texts(texts, model, tokenizer)

    # Speichere die Vorhersagen in eine neue CSV-Datei
    save_predictions_to_csv(df, predictions, output_csv)


# Beispielausführung
if __name__ == "__main__":
    input_csv = "input_sentiment_data.csv"  # Pfad zur Eingabe-CSV
    output_csv = "output_sentiment_predictions.csv"  # Pfad zur Ausgabe-CSV
    model_name = "./hugging_face_bert_sentiment/finetuned_models"  # Pfad zum finetunten Modell

    apply_model_to_csv(input_csv, output_csv, model_name)
