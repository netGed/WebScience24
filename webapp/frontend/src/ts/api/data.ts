import axios from "axios";
import { TTweetData } from "../../types.ts";

export const getEvaluationData = async () => {
  const baseUrl = "/api/get_evaluation_data/";

  const config = {
    method: "get",
    maxBodyLength: Infinity,
    url: baseUrl,
    headers: {},
  };

  return await axios
    .request(config)
    .then((response) => {
      const str = JSON.stringify(response.data);
      const data = JSON.parse(str);
      return data as TTweetData[];
    })
    .catch((error) => {
      console.log(error);
    });
};

export const getRandomTestDataOld = async () => {
  const baseUrl = "/api/get_random_test_data_old/?count=100";

  const config = {
    method: "post",
    maxBodyLength: Infinity,
    url: baseUrl,
    data: 100,
  };

  return await axios
    .request(config)
    .then((response) => {
      const str = JSON.stringify(response.data);
      const data = JSON.parse(str);
      return data as TTweetData[];
    })
    .catch((error) => {
      console.log(error);
    });
};

export const getRandomTestDataNew = async () => {
  const baseUrl = "/api/get_random_test_data_new/?count=100";

  const config = {
    method: "post",
    maxBodyLength: Infinity,
    url: baseUrl,
    data: 100,
  };

  return await axios
    .request(config)
    .then((response) => {
      const str = JSON.stringify(response.data);
      const data = JSON.parse(str);
      return data as TTweetData[];
    })
    .catch((error) => {
      console.log(error);
    });
};

export const getRandomTestDataMixed = async () => {
  const baseUrl = "/api/get_random_test_data_mixed/?count=100";

  const config = {
    method: "post",
    maxBodyLength: Infinity,
    url: baseUrl,
    data: 100,
  };

  return await axios
    .request(config)
    .then((response) => {
      const str = JSON.stringify(response.data);
      const data = JSON.parse(str);
      return data as TTweetData[];
    })
    .catch((error) => {
      console.log(error);
    });
};
