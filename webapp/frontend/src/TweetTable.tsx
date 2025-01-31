import React, { useEffect, useState } from "react";
import { Column } from "primereact/column";
import { observer } from "mobx-react-lite";
import { DataTable } from "primereact/datatable";
import Papa, { ParseResult } from "papaparse";
import { tweet_data } from "./assets/data.ts";
import { Button } from "primereact/button";
import axios from "axios";

type TweetData = {
  id: number;
  tweet: string;
  label: number;
};

const TweetTable: React.FC = () => {
  const [values, setValues] = useState<TweetData[] | undefined>();
  const [message, setMessage] = useState<string | null>(null);

  const loadData = () => {
    Papa.parse(tweet_data, {
      header: true,
      skipEmptyLines: true,
      delimiter: ",",
      complete: (results: ParseResult<TweetData>) => {
        setValues(results.data as TweetData[]);
        console.log("res:", results);
        console.log("data:", results.data);
      },
    });
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleClassificationClick = (rowData: TweetData) => {
    console.log(rowData["tweet"]);
    const tweet = rowData["tweet"];
    const baseUrl = "/api/gettest/";

    const data = new FormData();
    data.append("request_body", tweet);

    const config = {
      method: "post",
      maxBodyLength: Infinity,
      url: baseUrl, // "127.0.0.1:8000/gettest",
      headers: {},
      data: data,
    };

    axios
      .request(config)
      .then((response) => {
        console.log(JSON.stringify(response.data));
      })
      .catch((error) => {
        console.log(error);
      });
  };

  const classificationTemplate = (rowData: TweetData) => {
    return (
      <Button
        type="button"
        icon="pi pi-refresh"
        className="p-button-sm p-button-text"
        onClick={() => handleClassificationClick(rowData)}
      />
    );
  };

  return (
    <div className="card">
      <DataTable value={values} tableStyle={{ minWidth: "50rem" }}>
        <Column field="id" header="Id"></Column>
        <Column field="tweet" header="Tweet"></Column>
        <Column field="label" header="Label"></Column>
        <Column field="classify" header="Classify"></Column>
        <Column
          style={{ flex: "0 0 4rem" }}
          body={classificationTemplate}
        ></Column>
      </DataTable>
    </div>
  );
};

export default observer(TweetTable);
