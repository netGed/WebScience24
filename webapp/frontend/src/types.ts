export type TTweetData = {
  id: number;
  tweet: string;
  label: number;
  new_label: number;
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
