import axios from "axios";
import { TClassificationData, TPredictionData, TTweetData } from "../types.ts";

export const getPredictionEnsemble = async (tweet: string) => {
  const baseUrl = "/api/get_prediction_ensemble/";

  const data = new FormData();
  data.append("request_body", tweet);

  const config = {
    method: "post",
    maxBodyLength: Infinity,
    url: baseUrl,
    headers: {},
    data: data,
  };

  return await axios
    .request(config)
    .then((response) => {
      return response.data as TPredictionData;
    })
    .catch((error) => {
      console.log(error);
    });
};

export const getPredictions = async (tweet: string) => {
  const baseUrl = "/api/get_predictions/";

  const data = new FormData();
  data.append("request_body", tweet);

  const config = {
    method: "post",
    maxBodyLength: Infinity,
    url: baseUrl,
    headers: {},
    data: data,
  };

  return await axios
    .request(config)
    .then((response) => {
      return response.data as TPredictionData[];
    })
    .catch((error) => {
      console.log(error);
    });
};

export const getClassificationResults = async (tweets: TTweetData[]) => {
  const baseUrl = "/api/get_classifications/";
  console.log(tweets);

  const config = {
    method: "post",
    maxBodyLength: Infinity,
    url: baseUrl,
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    data: JSON.stringify(tweets),
  };

  return await axios
    .request(config)
    .then((response) => {
      return response.data as TClassificationData[];
    })
    .catch((error) => {
      console.log(error);
    });
};
