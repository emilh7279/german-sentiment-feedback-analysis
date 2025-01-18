from textblob_de import TextBlobDE as TextBlob

"""
Wenn Sie unsicher sind, welche Datensätze/Modelle Sie benötigen, können Sie das "popular"-Subset der NLTK-Daten installieren.
Geben Sie dazu in der Kommandozeile "python -m nltk.downloader popular" ein oder im Python-Interpreter:
import nltk; nltk.download('popular')

If you’re unsure of which datasets/models you’ll need, you can install the "popular" subset of NLTK data.
On the command line, type: python -m nltk.downloader popular, or in the Python interpreter:
import nltk; nltk.download('popular')

Das NLTK Toolkit (Natural Language Toolkit) muss beim ersten Ausführen des Codes auskommentiert werden.
The NLTK Toolkit (Natural Language Toolkit) needs to be uncommented during the first run of the code.
"""

# Importieren und Herunterladen von NLTK-Daten (bei Bedarf aktivieren).
# Import and download NLTK data (activate if needed).
# import nltk
# nltk.download('popular')
# nltk.download('punkt')

def satz_bewertung(satz):
    """
    Bewertet die Stimmung eines deutschen Satzes und gibt aus, ob diese positiv, negativ oder neutral ist.
    Analyzes the sentiment of a German sentence and returns whether it is positive, negative, or neutral.

    Parameter:
    satz (str): Der Satz, dessen Stimmung bewertet werden soll.
    satz (str): The sentence whose sentiment is to be analyzed.

    Rückgabe:
    str: Eine Nachricht mit der Stimmung und der Polarität des Satzes.
    str: A message containing the sentiment and polarity of the sentence.
    """

    # Erstellen eines TextBlob-Objekts
    # Create a TextBlob object
    blob = TextBlob(satz)

    # Analysieren der Polarität des Satzes
    # Analyze the polarity of the sentence
    polarität = blob.sentiment.polarity

    # Bestimmen der Stimmung basierend auf der Polarität
    # Determine sentiment based on polarity
    if polarität > 0:
        return f"Der Satz ist positiv. Polarität: {polarität}"
    elif polarität < 0:
        return f"Der Satz ist negativ. Polarität: {polarität}"
    else:
        return f"Der Satz ist neutral. Polarität: {polarität}"

# Testen der Funktion mit einigen Beispielsätzen
# Testing the function with some example sentences
sätze = [
    "Der Service war hervorragend und mein Anliegen wurde sofort geklärt.",
    "Der Prozess war etwas langwierig, aber das Endergebnis war in Ordnung.",
    "Der Mitarbeiter war sehr unhöflich und mein Problem wurde nicht behoben.",
    "Schlechter Kundenservice, niemand hat auf meine Anfragen geantwortet.",
    "Der Service hat zu lange gebraucht und mein Problem nicht gelöst.",
    "Ich bin wirklich beeindruckt, wie schnell mein Problem gelöst wurde.",
    "Fantastischer Service! Mein Problem wurde sofort gelöst.",
    "Es hat viel zu lange gedauert und niemand konnte mir helfen.",
    "Es war okay, nicht das Beste, aber das Problem wurde gelöst.",
    "Das Support-Team schien verwirrt und hat nicht wirklich geholfen."
]

# Ausgabe der Ergebnisse für jeden Satz
# Output the results for each sentence
for satz in sätze:
    print(satz_bewertung(satz))
