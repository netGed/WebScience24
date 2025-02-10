export type TTweetData = {
  id: number;
  tweet: string;
  label: number;
  new_label: number;
};

export type TTweetDataWithMetric = {
  id: number;
  tweet: string;
  label: number;
  ensemble_metric: number;
  nb_metric: number;
  svm_metric: number;
  gru_metric: number;
  lstm_metric: number;
  bert_metric: number;
  roberta_metric: number;
};

export type TPredictionData = {
  model_name: string;
  zero_proba: number;
  one_proba: number;
  label: number;
};

export type TClassificationData = {
  model_name: string;
  accuracy: number;
  f1_score: number;
  precision: number;
  recall: number;
};
