import os
import csv
import datetime as dt
from pathlib import Path
import pandas as pd
from textblob_de import TextBlobDE as TextBlob

def satz_bewertung(satz):
    """
    Bewertet die Stimmung eines deutschen Satzes (positiv, negativ oder neutral).
    Mithilfe des textblob_de Modul
    """
    # Erstellen eines TextBlob-Objekts
    blob = TextBlob(satz)

    # Polarität des Satzes analysieren
    polarität = blob.sentiment.polarity

    # Stimmungsbewertung basierend auf Polarität
    if polarität > 0:
        return "positiv", polarität
    elif polarität < 0:
        return "negativ", polarität
    else:
        return "neutral", polarität

def lade_saetze(dateiname):
    """
    Mithilfe der pandas Funktion read_csv wird das csv-File in ein pandas Dataframe
    geladen und die Spalte mit den enthaltenen Feedback-Texten in das List-Objekt
    saetze geschrieben. Das List-Objekt wird an die Funktion zurückgegeben.
    """
    arbeitsverzeichniss = Path.cwd()
    daten_verzeichnis = str(arbeitsverzeichniss.parent) + r"\input_data"
    # dateiname = 'feedback.csv'
    file_to_open = os.path.join(daten_verzeichnis, dateiname)
    df = pd.read_csv(file_to_open, delimiter=";")
    saetze = df['Feedback_Text'].tolist()
    return saetze

def analysiere_saetze(saetze):
    """
    Funktion: Analysiert eine Liste von Sätzen und gibt die Ergebnisse zurück.
    iteriert über die Sätze aus der Liste / pandas Dataframe und übergibt sie an Funktion
    satz_bewertung(). Das List objekt ergebnisse wird bei jeder Iteration erweitert
    """
    ergebnisse = []
    for satz in saetze:
        stimmung, polarität = satz_bewertung(satz)
        ergebnisse.append((satz, stimmung, polarität))
    return ergebnisse

def speichere_ergebnisse(ergebnisse):
    """
    Funktion: Speichert die Ergebnisse der Analyse in einer CSV-Datei.
    """
    # Deklariere Dateinamen-Variablen
    datum = dt.datetime.now().strftime("%Y%m%d_%HT%M")
    result_pfad = "results/"
    dateiname = result_pfad + str(datum) + "_Ergebnisse.csv"

    # Schreibe CSV Ergebiss-Datei
    with open(dateiname, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Satz", "Stimmung", "Polarität"])
        for satz, stimmung, polarität in ergebnisse:
            writer.writerow([satz, stimmung, polarität])

# Lade Sätze
saetze =  lade_saetze("feedback.csv")

# Sätze analysieren
ergebnisse = analysiere_saetze(saetze)

# Ergebnisse speichern
speichere_ergebnisse(ergebnisse)

print("Analyse abgeschlossen und Ergebnisse gespeichert.")
