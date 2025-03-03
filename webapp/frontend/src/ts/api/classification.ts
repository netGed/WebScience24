import axios from "axios";
import {
  TClassificationData,
  TClassificationDataWithMetrics,
  TClassificationModelData,
  TTweetData,
} from "../../types.ts";

export const getClassificationForTweet = async (tweet: string) => {
  const baseUrl = "/api/get_classification_data/";
  // const data = new FormData();
  // data.append("request_body", tweet);

  const config = {
    method: "post",
    maxBodyLength: Infinity,
    url: baseUrl,
    headers: {
      "Content-Type": "application/json",
    },
    data: JSON.stringify({tweet})
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

export const getClassificationForTweetAlt = async (tweet: string) => {
  const baseUrl = "/api/get_classification_data_alt/";
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
      return response.data as TClassificationModelData;
    })
    .catch((error) => {
      console.log(error);
    });
};

export const getClassificationMetrics = async (tweets: TTweetData[]) => {
  console.log("tweets", tweets);
  const baseUrl = "/api/get_classification_metrics/";
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
      return response.data as TClassificationDataWithMetrics[];
    })
    .catch((error) => {
      console.log(error);
    });
};
