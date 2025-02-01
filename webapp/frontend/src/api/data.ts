import axios from "axios";
import { TTweetData } from "../types.ts";

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
      console.log("Response: ", response);
      const str = JSON.stringify(response.data);
      const data = JSON.parse(str);
      return data as TTweetData[];
    })
    .catch((error) => {
      console.log(error);
    });
};
