import os
import csv
import datetime as dt
from pathlib import Path
import pandas as pd
from textblob_de import TextBlobDE as TextBlob

def sentence_classification(sentence):
    """
    Bewertet die Stimmung eines deutschen Satzes (positiv, negativ oder neutral).
    Mithilfe des textblob_de Modul
    """
    # Erstellen eines TextBlob-Objekts
    blob = TextBlob(sentence)

    # Polarität des Satzes analysieren
    polarity = blob.sentiment.polarity

    # Stimmungsbewertung basierend auf polarity
    if polarity > 0:
        return "positiv", polarity
    elif polarity < 0:
        return "negativ", polarity
    else:
        return "neutral", polarity

def load_sentences(filename):
    """
    Mithilfe der pandas Funktion read_csv wird das csv-File in ein pandas Dataframe
    geladen und die Spalte mit den enthaltenen Feedback-Texten in das List-Objekt
    saetze geschrieben. Das List-Objekt wird an die Funktion zurückgegeben.
    """
    working_dir = Path.cwd()
    data_dir = str(working_dir.parent) + r"\input_data"
    file_to_open = os.path.join(data_dir, filename)
    df = pd.read_csv(file_to_open, delimiter=";")
    sentences = df['Feedback_Text'].tolist()
    return sentences

def analyze_sentences(sentences):
    """
    Funktion: Analysiert eine Liste von Sätzen und gibt die Ergebnisse zurück.
    iteriert über die Sätze aus der Liste / pandas Dataframe und übergibt sie an Funktion
    satz_bewertung(). Das List objekt ergebnisse wird bei jeder Iteration erweitert
    """
    results = []
    for sentence in sentences:
        mood, polarity = sentence_classification(sentence)
        results.append((sentence, mood, polarity))
    return results

def save_results(results):
    """
    Funktion: Speichert die Ergebnisse der Analyse in einer CSV-Datei.
    """
    # Deklariere Dateinamen-Variablen
    datum = dt.datetime.now().strftime("%Y%m%d_%HT%M")
    result_pfad = "results/"
    filename = result_pfad + str(datum) + "_Ergebnisse.csv"

    # Schreibe CSV Ergebiss-Datei
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["sentence", "mood", "polarity"])
        for sentence, mood, polarity in results:
            writer.writerow([sentence, mood, polarity])

# Lade Sätze
sentences =  load_sentences("feedback.csv")

# Sätze analysieren
results = analyze_sentences(sentences)

# Ergebnisse speichern
save_results(results)

print("Analyse abgeschlossen und Ergebnisse gespeichert.")
