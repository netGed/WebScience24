export type TTweetData = {
  id: number;
  tweet: string;
  label: number;
};

export type TTweetDataWithClassifications = TTweetData &
  TClassificationModelData;

export type TClassificationData = {
  model_name: string;
  zero_probability: number;
  one_probability: number;
  label: number;
};

export type TClassificationDataWithMetrics = {
  model_name: string;
  accuracy: number;
  f1_score: number;
  precision: number;
  recall: number;
};

export type TClassificationModelData = {
  classification_ensemble?: number;
  classification_svm?: number;
  classification_nb?: number;
  classification_gru?: number;
  classification_lstm?: number;
  classification_bert?: number;
  classification_roberta?: number;
};
