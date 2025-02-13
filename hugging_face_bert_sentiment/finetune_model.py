import os
from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

# Lade den CSV-Datensatz
arbeitsverzeichniss = Path.cwd()                                                            # Ermittle aktuelles Arbeitsverzeichnis
daten_verzeichnis = str(arbeitsverzeichniss.parent) + r"\input_data\bewertete_daten"        # Finde Datenverzeichnis
dateiname = "Mappe1.csv"                                                                      # Definiere Dateiname Trainingsdaten
file_to_open = os.path.join(daten_verzeichnis, dateiname)                                   # Kombiniere Datenverzeichis und Dateiname
df = pd.read_csv(file_to_open, delimiter=";")
dataset = Dataset.from_pandas(df)                                                           # Konvertiere Pandas Dataframe in Hugging Face Dataset

# Lade den Tokenizer
model_name = "oliverguhr/german-sentiment-bert"                                             # Definiere Model (german pretrained Model oliver Guhr)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenisiere den Datensatz
def tokenize_function(examples):
    return tokenizer(examples['Text'], padding="max_length", truncation=True)               # Trainingsdatei muss Spalte mit Überschrift Text enthalten

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Aufteilen des Datensatzes in Trainings- und Testdaten
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Lade das Modell
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Konfiguriere die Trainingsparameter
training_args = TrainingArguments(
    output_dir="finetuned_hugging_face_sentiment/results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialisiere den Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Trainiere das Modell
trainer.train()

# Evaluierung des Modells
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Speichern des finetunten Modells
model.save_pretrained("hugging_face_bert_sentiment/finetuned_models")
tokenizer.save_pretrained("hugging_face_bert_sentiment/finetuned_models")
