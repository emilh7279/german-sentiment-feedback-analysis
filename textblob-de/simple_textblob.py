from textblob_de import TextBlobDE as TextBlob

"""
If you’re unsure of which datasets/models you’ll need, you can install the “popular” subset of NLTK data,
on the command line type python -m nltk.downloader popular, or in the Python interpreter import nltk; nltk.download('popular')

NLTK Toolkit Natural Language Toolkit muss vor dem beim ersten Ausführen des Codes auskommentiert werden.
"""

#import nltk
#nltk.download('popular')
#nltk.download('punkt')

def satz_bewertung(satz):

    # Bewertet die Stimmung eines deutschen Satzes (positiv, negativ oder neutral).
    #

    # Erstellen eines TextBlob-Objekts
    # Create textblob objekt
    blob = TextBlob(satz)

    # Polarität des Satzes analysieren
    # Analyse polarity
    polarität = blob.sentiment.polarity

    # Stimmungsbewertung basierend auf Polarität
    # Sentiment and Polarity
    if polarität > 0:
        return f"Der Satz ist positiv. Polarität: {polarität}"
    elif polarität < 0:
        return f"Der Satz ist negativ. Polarität: {polarität}"
    else:
        return f"Der Satz ist neutral. Polarität: {polarität}"


# Testen mit einigen Beispielsätzen
# Testing sentences
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

for satz in sätze:
    print(satz_bewertung(satz))
