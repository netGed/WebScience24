import axios from "axios";
import { TPredictionData } from "../types.ts";

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
