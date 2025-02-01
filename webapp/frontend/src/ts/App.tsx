import React, { useEffect, useState } from "react";
import "primeflex/primeflex.css";
import "primereact/resources/themes/lara-light-indigo/theme.css";
import "primeicons/primeicons.css";
import { Panel } from "primereact/panel";
import { Column } from "primereact/column";
import { DataTable } from "primereact/datatable";
import { Button } from "primereact/button";
import { TPredictionData, TSelectedTweet, TTweetData } from "../types.ts";
import { getPredictions } from "../api/predictions.ts";
import { classNames } from "primereact/utils";
import { InputText } from "primereact/inputtext";
import { Dropdown, DropdownChangeEvent } from "primereact/dropdown";
import { getEvaluationData } from "../api/data.ts";

const labels = [{ label: 0 }, { label: 1 }];

const App: React.FC = () => {
  const [tweetData, setTweetData] = useState<TTweetData[] | undefined>();
  const [selectedTweet, setSelectedTweet] = useState<TSelectedTweet>();
  const [predictionData, setPredictionData] = useState<TPredictionData[]>([]);
  const [loading, isLoading] = useState(false);
  const [tweetText, setTweetText] = useState<string>("");
  const [tweetLabel, setTweetLabel] = useState(labels[0]);

  const isInputInvalid = () => {
    return tweetText.length < 1;
  };

  useEffect(() => {
    const loadDataAsync = async () => {
      const data = await getEvaluationData();
      console.log(data);
      setTweetData(data as TTweetData[]);
    };

    loadDataAsync();
  }, []);

  const handleClassificationClick = async (tweet: string, label: number) => {
    isLoading(true);
    setSelectedTweet({ tweet: tweet, label: label });
    const result = (await getPredictions(tweet)) as TPredictionData[];
    console.log(result);
    setPredictionData(result);
    isLoading(false);
  };

  const classificationTemplate = (rowData: TTweetData) => {
    return (
      <Button
        type="button"
        icon="pi pi-tags"
        className="p-button-sm p-button-text"
        onClick={() =>
          handleClassificationClick(rowData["tweet"], rowData["label"])
        }
      />
    );
  };

  const predictionTemplate = (rowData: TPredictionData) => {
    if (selectedTweet) {
      const realLabel = selectedTweet.label;

      const classColor = classNames(
        "border-circle w-2rem h-2rem inline-flex font-bold justify-content-center align-items-center text-sm",
        {
          "bg-red-100 text-red-900": rowData.label != realLabel,
          "bg-green-100 text-green-900": rowData.label == realLabel,
        },
      );

      return <div className={classColor}>{rowData.label}</div>;
    }
  };

  return (
    <>
      <Panel header="Praktikum WebScience">
        <div className="flex">
          <div className="flex flex-column m-2" style={{ width: "50%" }}>
            <div className="flex">
              <DataTable
                value={tweetData}
                tableStyle={{ minWidth: "50rem" }}
                scrollable
                scrollHeight="50rem"
              >
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
            <div className="flex mt-2">
              <div className="ml-2 flex flex-row align-items-center">
                <h3>Tweet: </h3>
                <InputText
                  className="m-2"
                  style={{ width: "45rem" }}
                  id="tweetInput"
                  value={tweetText}
                  onChange={(e) => setTweetText(e.target.value)}
                  invalid={isInputInvalid()}
                />
                <h3>Label: </h3>
                <Dropdown
                  value={tweetLabel}
                  onChange={(e: DropdownChangeEvent) => setTweetLabel(e.value)}
                  options={labels}
                  optionLabel="label"
                  defaultValue={"0"}
                  className="w-full md:w-5rem m-2"
                />
                <Button
                  type="button"
                  icon="pi pi-tags"
                  className="p-button-sm p-button-text"
                  disabled={isInputInvalid()}
                  onClick={() =>
                    handleClassificationClick(tweetText, tweetLabel.label)
                  }
                />
              </div>
            </div>
          </div>

          <div className="flex flex-column m-2" style={{ width: "50%" }}>
            <div className="ml-2 flex flex-row align-items-center">
              <h3>Ausgewählter Tweet: </h3>
              <div className="ml-2">
                {selectedTweet ? selectedTweet["tweet"] : ""}
              </div>
            </div>
            <div className="ml-2 flex flex-row align-items-center">
              <h3>Label: </h3>
              <div className="ml-2">
                {selectedTweet ? selectedTweet["label"] : ""}
              </div>
            </div>

            <DataTable
              value={predictionData}
              tableStyle={{ minWidth: "50rem" }}
              scrollable
              scrollHeight="50rem"
              loading={loading}
            >
              <Column field="model_name" header="Model Name"></Column>
              <Column
                field="zero_proba"
                header="No Hatespeech-Probability"
              ></Column>
              <Column
                field="one_proba"
                header="Hatespeech-Probability"
              ></Column>
              <Column
                field="label"
                header="Prediction"
                body={predictionTemplate}
              ></Column>
            </DataTable>
          </div>
        </div>
      </Panel>
    </>
  );
};

export default App;
