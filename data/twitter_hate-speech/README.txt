Beschreibung der Datensätze
------------------------------------------------------------------------------------------------------------------------

train.csv
- entspricht 70% des ursprünglichen TRAIN-Datensatzes
- keine Vorverarbeitungsschritte ausgeführt

train_cleaned.csv
- train.csv in bereinigter Form (alle Bereinigungs- und Vorverarbeitungsschritte)
- soll für das Modelltraining verwendet werden

train_cleaned_rnn.csv
- train.csv in bereinigter Form (alle Bereinigungs- und Vorverarbeitungsschritte AUßER: Löschen von Zeilen, Stopwords, Most/Least frequent words)
- soll für das Modelltraining von RNNs verwendet werden

train_cleaned_rnn-basic.csv
- train.csv, nur Encoding-Fix und Duplikatentfernung
- soll für das Modelltraining von RNNs verwendet werden

------------------------------------------------------------------------------------------------------------------------

test.csv
- entspricht den übrigen 30% des ursprünglichen TRAIN-Datensatzes
- keine Vorverarbeitungsschritte ausgeführt

test_cleaned.csv
- test.csv in bereinigter Form (alle Bereinigungs- und Vorverarbeitungsschritte AUßER Löschen von Zeilen)
- soll nur für die Modellevaluation verwendet werden

test_cleaned_rnn.csv
- test.csv in bereinigter Form (alle Bereinigungs- und Vorverarbeitungsschritte AUßER: Löschen von Zeilen, Stopwords, Most/Least frequent words)
- soll nur für die Modellevaluation von RNNs verwendet werden

test_cleaned_rnn-basic.csv
- test.csv, nur Encoding-Fix und Duplikatentfernung
- soll nur für die Modellevaluation von RNNs verwendet werden

------------------------------------------------------------------------------------------------------------------------

predict.csv
- entspricht dem ursprünglichen TEST-Datensatz
- keine Vorverarbeitungsschritte ausgeführt!

predict_cleaned.csv
- predict.csv in bereinigter Form (alle Bereinigungs- und Vorverarbeitungsschritte AUßER Löschen von Zeilen)

predict_cleaned_rnn.csv
- predict.csv in bereinigter Form (alle Bereinigungs- und Vorverarbeitungsschritte AUßER: Löschen von Zeilen, Stopwords, Most/Least frequent words)

predict_cleaned_rnn-basic.csv
- predict.csv, nur Encoding-Fix und Duplikatentfernung