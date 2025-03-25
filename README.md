# Klassifikation von Hate Speech am Beispiel von Twitter-Nachrichten

In diesem Repository werden die Ergebnisse des WebScience-Praktikums 2024/2025 festgehalten.

## Daten
+ [twitter_hate-speech](https://github.com/netGed/WebScience24/tree/main/data/twitter_hate-speech): Ursprüngliche Daten, für die weitere Verarbeitung der einzelnen Modelle und nach Train/Test/Predict aufgeteilt. Bei Predict handelt es sich um Daten ohne Label (wurden nicht bereitgestellt), daher können diese Daten nur für eine stichprobenartige Klassifikation verwendet werden.
+ [new_datasets](https://github.com/netGed/WebScience24/tree/main/data/new_datasets): Enthält die erweiterten Datensätze
+ [mixed-data](https://github.com/netGed/WebScience24/tree/main/data/mixed_dataset): Enthält die fusionierten Datensätze aus twitter_hate-speech und new_datasets, aufgeteilt nach Train/Test. Test soll dabei für die finale Modellevaluation verwendet werden, innerhalb des Modelltrainings muss Train daher erneut in Train/Test aufgeteilt werden.
+ [manual_labeled](https://github.com/netGed/WebScience24/tree/main/data/manual_labeled): Stichprobe der mixed-data-Testdaten, bei denen das Label durch die Gruppe kontrolliert wurde.
+ Die Jupyter-Notebooks im [data](https://github.com/netGed/WebScience24/tree/main/data)-Ordner sind für das Data-Splitting und -Preprocessing (=> Aufruf der einzelnen Aufbereitungsschritte gem. Anforderungen) zuständig. Bei den Datensätzen für Deep-Learning-Modelle wurden z.B. Lemmatisierung, Entfernen von Stopwords nicht ausgeführt. 

## Explorative Datenanalyse
+ Das Jupyter-Notebook im ([Data Exploring](https://github.com/netGed/WebScience24/blob/accfb11e2146db2f377ce93ae31b101dbe01051a/src/1.%20Explorative%20Analysis%20%26%20Preprocessing/team/1_2_explore_data.ipynb)) wurden die allgemeine Data Explorationsschritte und im Notebook [Data Qualitiy Checks](https://github.com/netGed/WebScience24/blob/accfb11e2146db2f377ce93ae31b101dbe01051a/src/1.%20Explorative%20Analysis%20%26%20Preprocessing/team/0_0_data_quality_check.ipynb) wurden die Datenqualität Checks durchgeführt.
  
## Datenvorverarbeitung

### Datenbereinigung
Die Funktionen zur Datenbereinigung wurden im zentralen Ordner [functions/cleaning_functions](https://github.com/netGed/WebScience24/blob/accfb11e2146db2f377ce93ae31b101dbe01051a/src/functions/clean_data_generic_functions.py) abgelegt. 
Diese wurden im Notebook [Data Cleaning Notebook](https://github.com/netGed/WebScience24/blob/accfb11e2146db2f377ce93ae31b101dbe01051a/src/1.%20Explorative%20Analysis%20%26%20Preprocessing/team/1_1_clean_data.ipynb) verwendet, um initialen Datensatz durch erstellten Funktionen zu bereinigen. 

### Resampling 
Verschiedene Teammitglieder haben bereits zu Beginn der Trainingsphase 1 mit der Implementierung geeigneter Resampling-Methoden begonnen. Die dabei entstandenen Ergebnisse sind unter [resampling] (https://github.com/netGed/WebScience24/tree/main/src/1a.%20Resampling%20Methods) einsehbar. Nachdem sich jedoch die Datenqualität des ursprünglichen Datensatzes als unzureichend erwiesen hatte, wurde diese Arbeit gestoppt. Stattdessen wurde der Fokus auf das Training mit den erweiterten Daten und die Implementierung zusätzlicher Modelle gelegt.

### Vektorisierung
Die Vektorisierungsfunktionen wurden im zentralen Ordner [functions/vectorizing](https://github.com/netGed/WebScience24/blob/accfb11e2146db2f377ce93ae31b101dbe01051a/src/functions/vectorize_functions.py) abgelegt. Jedoch haben einige Modelle diese Funktionen speziell für diese Modelle angepasst. (siehe [LSTM Glove](https://github.com/netGed/WebScience24/blob/c858a5a7f404ab17090bdaed5d69d01615f158b7/src/3.%20Deep%20Learning%20Approach/nasiba/Training/Phase_2/vector_functions.py). Die Funktionen wurden direkt im jeweiligen Modelltraining in Notebooks als Datenvorbereitungsschritt angewendet, bevor diese in Trainings eingespeist wurden.

## Modelltraining

### Ensemble
+ [01_olddata_ensemble_training_tuning_model-selection.ipynb](https://github.com/netGed/WebScience24/tree/main/src/2.%20Classical%20ML%20Methods/chris): Notebook für die Trainingsphase 1 von Ensemble-Modellen auf den ursprünglichen Daten. Hierbei wurde zuerst eine geeignete Vektorisierungsart als Basis bestimmt (TF-IDF). Anschließend wurden verschiedene Modelle aus Libraries wie CatBoost, XGBoost oder (Balanced-)RandomForest trainiert und ausgewertet. Für das beste Modell dieser Phase wurde anschließend ein umfassendes Hyperparametertuning (Parameter, Scoring) durchgeführt.  
+ [02_mixed_ensemble_training_tuning_model-selection.ipynb](https://github.com/netGed/WebScience24/tree/main/src/2.%20Classical%20ML%20Methods/chris): Notebook analog zum Vorgehen für die ursprünglichen Daten, hier dann Trainingsphase 2 auf den erweiterten Daten (mixed-data). 
+ [03_mixed_ensemble_detail-evaluation.ipynb](https://github.com/netGed/WebScience24/blob/main/src/2.%20Classical%20ML%20Methods/chris/03_mixed_ensemble_detail-evaluation.ipynb): Detaillierte Auswertung der besten beiden Modelle (BalancedRF und CatBoostClassifier), dabei Analyse, welche Tweets falsch vorhergesagt werden und was sie gemeinsam haben bzw. wo sie sich unterscheiden.
### Naive Bayes

### Support Vector Machine
- zu finden im Ordner "Elena"
- 1_original_data enthält die Dateien aus Trainingsphase 1, in welcher auf dem initalen Datensatz trainiert wurde
- In Trainingsphase 1 wurden für die drei Vektorisierungsmethoden
  - [glove](https://github.com/netGed/WebScience24/blob/main/src/2.%20Classical%20ML%20Methods/elena/1_original_data/glove/SVM_glove_grid_original_data.ipynb)
  - [tfidf](https://github.com/netGed/WebScience24/blob/main/src/2.%20Classical%20ML%20Methods/elena/1_original_data/tfidf/SVM_tfidf_grid_original_data.ipynb)
  - [w2v](https://github.com/netGed/WebScience24/blob/main/src/2.%20Classical%20ML%20Methods/elena/1_original_data/w2v/SVM_w2v_grid_original_data.ipynb)
    
  jeweils verschiedene SVM Konfigurationen für die vier Kernel linear, sigmoid, rbf und poly trainiert
- Die [Evaulation](https://github.com/netGed/WebScience24/blob/main/src/2.%20Classical%20ML%20Methods/elena/1_original_data/evaluate_models.ipynb) der Trainingsphase 1 fand über alle Outputs hinweg statt. 
  
- Mixed_data enhätlt die Dateien aus Trainingsphase 2, in welcher das Training auf dem erweiterten Datensatz durchgeführt wurde
- In beiden Trainingsphasen wurde jeweils für die drei Vektorisierungsmethoden
  - [glove](https://github.com/netGed/WebScience24/blob/main/src/2.%20Classical%20ML%20Methods/elena/mixed_data/glove/B_Lauf_neue_reports/svm_glove_optimized.ipynb)
  - [tfidif](https://github.com/netGed/WebScience24/blob/main/src/2.%20Classical%20ML%20Methods/elena/mixed_data/tfidf/3_Lauf_neue_reports/svm_tfidf_optimized.ipynb)
  - [w2v](https://github.com/netGed/WebScience24/blob/main/src/2.%20Classical%20ML%20Methods/elena/mixed_data/w2v/3_Lauf_neue_reports/svm_w2v_optimized.ipynb)
  
  jeweils verschiedene SVM Konfigurationen für die vier Kernel linear, sigmoid, rbf und poly trainiert
- Die [Evaluation](https://github.com/netGed/WebScience24/blob/main/src/2.%20Classical%20ML%20Methods/elena/mixed_data/evaluate_models_new_reports.ipynb) der Trainingsphase 2 fand über alle Outputs hinweg statt.

### RNN-LSTM
LSTM-Modelle wurden im Ordner [LSTM Trainings](https://github.com/netGed/WebScience24/tree/accfb11e2146db2f377ce93ae31b101dbe01051a/src/3.%20Deep%20Learning%20Approach/nasiba/Training) für 2 Phasen in jeweiligen Ordner [Phase 1](https://github.com/netGed/WebScience24/tree/accfb11e2146db2f377ce93ae31b101dbe01051a/src/3.%20Deep%20Learning%20Approach/nasiba/Training/Phase_1) und [Phase 2](https://github.com/netGed/WebScience24/tree/accfb11e2146db2f377ce93ae31b101dbe01051a/src/3.%20Deep%20Learning%20Approach/nasiba/Training/Phase_2) trainiert. 
Die Trainings und Ergebnisse wurden für jede Vektorisierungsart jeweils separat gespeichert.

### RNN-GRU
+ [deep-learning-models.ipynb](https://github.com/netGed/WebScience24/blob/main/src/3.%20Deep%20Learning%20Approach/imran/deep-learning-models.ipynb): Notebook für das Training der GRU-Modelle, das sowohl Phase 1 als auch Phase 2 abdeckt. Da die Hyperparameter-Optimierung in Phase 2 keine weiteren Leistungsverbesserungen erzielte, wurde die Architektur sowie die Parameterkonfiguration aus Phase 1 übernommen. Für die Modellentwicklung wurden verschiedene Varianten mit unterschiedlichen Embedding-Methoden und Parametereinstellungen trainiert. Final ausgewählt wurde die Konfiguration, die die beste Modellperformance erreichte.

### BERT
[bert-models.ipynb](https://github.com/netGed/WebScience24/blob/main/src/3.%20Deep%20Learning%20Approach/imran/bert-models.ipynb): Notebook für das Finetuning von BERT-Modellen.. Aufgrund der hohen Rechenanforderungen wurde das Training ausschließlich auf der GPU-Infrastruktur der Universität durchgeführt. Je nach Experiment wurden im Notebook flexibel Parameter angepasst, darunter das eingesetzte Basismodell [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased), die Anzahl der Epochen sowie der Scoring-Mechanismus. Dank des modularen Aufbaus konnte das gesamte Finetuning verschiedener BERT-Modelle zentral in einem einzigen Notebook umgesetzt werden.

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
