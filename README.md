# Klassifikation von Hate Speech am Beispiel von Twitter-Nachrichten

In diesem Repository werden die Ergebnisse des WebScience-Praktikums 2024/2025 festgehalten.

## Daten
+ [twitter_hate-speech](https://github.com/netGed/WebScience24/tree/main/data/twitter_hate-speech): Ursprüngliche Daten, für die weitere Verarbeitung der einzelnen Modelle und nach Train/Test/Predict aufgeteilt. Bei Predict handelt es sich um Daten ohne Label (wurden nicht bereitgestellt), daher können diese Daten nur für eine stichprobenartige Klassifikation verwendet werden.
+ [new_datasets](https://github.com/netGed/WebScience24/tree/main/data/new_datasets): Enthält die erweiterten Datensätze
+ [mixed-data](https://github.com/netGed/WebScience24/tree/main/data/mixed_dataset): Enthält die fusionierten Datensätze aus twitter_hate-speech und new_datasets, aufgeteilt nach Train/Test. Test soll dabei für die finale Modellevaluation verwendet werden, innerhalb des Modelltrainings muss Train daher erneut in Train/Test aufgeteilt werden.
+ [manual_labeled](https://github.com/netGed/WebScience24/tree/main/data/manual_labeled): Stichprobe der mixed-data-Testdaten, bei denen das Label durch die Gruppe kontrolliert wurde.
+ Die Jupyter-Notebooks im [data](https://github.com/netGed/WebScience24/tree/main/data)-Ordner sind für das Data-Splitting und -Preprocessing (=> Aufruf der einzelnen Aufbereitungsschritte gem. Anforderungen) zuständig. Bei den Datensätzen für Deep-Learning-Modelle wurden z.B. Lemmatisierung, Entfernen von Stopwords nicht ausgeführt. 



## Explorative Datenanalyse

## Datenvorverarbeitung

### Resampling

### Vektorisierung

## Modelltraining

### Ensemble
+ [01_olddata_ensemble_training_tuning_model-selection.ipynb](https://github.com/netGed/WebScience24/tree/main/src/2.%20Classical%20ML%20Methods/chris): Notebook für die Trainingsphase 1 von Ensemble-Modellen auf den ursprünglichen Daten. Hierbei wurde zuerst eine geeignete Vektorisierungsart als Basis bestimmt (TF-IDF). Anschließend wurden verschiedene Modelle aus Libraries wie CatBoost, XGBoost oder (Balanced-)RandomForest trainiert und ausgewertet. Für das beste Modell dieser Phase wurde anschließend ein umfassendes Hyperparametertuning (Parameter, Scoring) durchgeführt.  
+ [02_mixed_ensemble_training_tuning_model-selection.ipynb](https://github.com/netGed/WebScience24/tree/main/src/2.%20Classical%20ML%20Methods/chris): Notebook analog zum Vorgehen für die ursprünglichen Daten, hier dann Trainingsphase 2 auf den erweiterten Daten (mixed-data). 
+ [03_mixed_ensemble_detail-evaluation.ipynb](https://github.com/netGed/WebScience24/blob/main/src/2.%20Classical%20ML%20Methods/chris/03_mixed_ensemble_detail-evaluation.ipynb): Detaillierte Auswertung der besten beiden Modelle (BalancedRF und CatBoostClassifier), dabei Analyse, welche Tweets falsch vorhergesagt werden und was sie gemeinsam haben bzw. wo sie sich unterscheiden.
### Naive Bayes

### Support Vector Machine

### RNN-GRU

### RNN-LSTM

### BERT

### RoBERTa
+ [01_finetuning_roberta-models.ipynb](https://github.com/netGed/WebScience24/blob/main/src/3.%20Deep%20Learning%20Approach/chris/01_finetuning_roberta-models.ipynb): Notebook für das Finetuning von RoBERTa-Modellen. Aufgrund der benötigten Rechenleistung fand das Training ausschließlich auf Hardware der Uni statt und es wurden je nach Modell Parameter im Notebook ausgetauscht (z.B. Basismodelle [roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), [roberta-base-hate](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate), Anzahl der Epochen, Scorer etc.), dementsprechend nur ein Notebook für das gesamte Finetuning.
+ [02_model_comparison.ipynb](https://github.com/netGed/WebScience24/blob/main/src/3.%20Deep%20Learning%20Approach/chris/02_model_comparison.ipynb): Vergleich der unterschiedlichen BERT/RoBERTa-Modelle. Für einen besseren Vergleich wurden hier auch die Basismodelle (außer BERT) für eine Bewertung des Finetunings betrachtet. 
+ [03_detail-evaluation.ipynb](https://github.com/netGed/WebScience24/blob/main/src/3.%20Deep%20Learning%20Approach/chris/03_detail-evaluation.ipynb): Detaillierte Auswertung der nachtrainierten RoBERTa-Modelle (hate/sentiment) und Betrachtung der falsch vorhergesagten Tweets. Ergebnisse grundsätzlich vergleichbar mit der Detailauswertung der Ensemble-Modelle.

## Modellevaluation
+ [01_evaluation_prediction.ipynb](https://github.com/netGed/WebScience24/blob/main/src/4.%20Evaluation/01_evaluation_prediction.ipynb): Gesamtauswertung aller Modelle auf einer Vielzahl von Datensätzen und mit unterschiedlichen Schwellwerten (einige Modelle wurden auf veränderten Schwellwerten trainiert). In diesem Notebook werden nur die Vorhersagen erstellt, in Dataframes gespeichert und als CSV abgelegt.
+ [01_evaluation_loaded.ipynb](https://github.com/netGed/WebScience24/blob/main/src/4.%20Evaluation/01_evaluation_loaded.ipynb): Lädt die CSV-Dateien für eine weiterführende Analyse, dabei:
  + Abschnitt 3: allgemeine Betrachtung der Metriken (F1-Score, Recall usw.), Vergleich und Visualisierung; Prüfung, ob es grundlegende Unterschiede bei der Vorhersage von unvorbereiteten Daten gibt und wie sich andere Schwellwerte auswirken.
  + Abschnitt 4: Detaillbetrachtung von falsch vorhergesagten Tweets auf den allgemeinen Testdaten und auf den manuell gelabelten Daten, erneut getrennt nach Schwellwert.
  + Abschnitt 5: Vergleich der manuell gelabelten Daten mit dem ursprünglichen Label

## Web-Applikation

### Frontend
+ React/Typescript WebApp, welche über REST-Anfragen die verschiedenen Datensätze und die Klassifikation von einzelnen Tweets anfragt. 

### Backend
+ Python-Backend mit FastAPI, welche die Anfragen des Frontends entgegennimmt und für die Klassifikation die Modelle aufruft. Aufgrund der Github-Größenbeschränkung können nicht alle Modelle gepusht werden.
