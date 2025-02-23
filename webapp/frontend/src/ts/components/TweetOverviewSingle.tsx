import React, { useState } from "react";
import { Panel } from "primereact/panel";
import { Column } from "primereact/column";
import { DataTable } from "primereact/datatable";
import { Button } from "primereact/button";
import { TClassificationData, TTweetData } from "../../types.ts";
import { getClassificationForTweet } from "../api/classification.ts";
import { classNames } from "primereact/utils";
import { InputText } from "primereact/inputtext";
import { Dropdown, DropdownChangeEvent } from "primereact/dropdown";
import { getRandomTestDataMixed } from "../api/data.ts";
import TweetStore from "../stores/TweetStore.ts";
import { observer } from "mobx-react-lite";

const labels = [{ label: 0 }, { label: 1 }];

const TweetOverviewSingle: React.FC = () => {
  const [selectedTweet, setSelectedTweet] = useState<TTweetData>();
  const [predictionData, setPredictionData] = useState<TClassificationData[]>(
    [],
  );
  const [loadingPrediction, isLoadingPrediction] = useState(false);
  const [tweetText, setTweetText] = useState<string>("");
  const [tweetLabel, setTweetLabel] = useState(labels[0]);

  const isInputInvalid = () => {
    return tweetText.length < 1;
  };

  const handleClassificationClickSingle = async (
    id: number,
    tweet: string,
    label: number,
  ) => {
    isLoadingPrediction(true);
    const tweetData = {
      id: id,
      tweet: tweet,
      label: label,
    };
    setSelectedTweet(tweetData);

    const predictionResult = (await getClassificationForTweet(
      tweet,
    )) as TClassificationData[];
    setPredictionData(predictionResult);

    isLoadingPrediction(false);
  };

  const handleRandomDataClick = async () => {
    const data = (await getRandomTestDataMixed()) as TTweetData[];
    if (data.length > 0) {
      setTweetText(data[0].tweet);
      setTweetLabel({ label: data[0].label });
    }
  };

  const classificationTemplate = (rowData: TTweetData) => {
    return (
      <Button
        rounded
        type="button"
        icon="pi pi-tags"
        className="p-button-sm p-button-text"
        onClick={() => {
          handleClassificationClickSingle(
            rowData["id"],
            rowData["tweet"],
            rowData["label"],
          );
        }}
      />
    );
  };

  const predictionTemplate = (rowData: TClassificationData) => {
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
      <div className="flex overflow-hidden" style={{ height: "85vh" }}>
        <div className="flex flex-column m-2" style={{ width: "50vw" }}>
          <Panel header="Tweetübersicht">
            <div className="flex">
              <DataTable
                value={TweetStore.tweets}
                scrollable
                scrollHeight="50rem"
                loading={TweetStore.loading}
              >
                <Column field="id" header="Id"></Column>
                <Column field="tweet" header="Tweet"></Column>
                <Column field="label" header="Label"></Column>
                <Column field="classify" header=""></Column>
                <Column body={classificationTemplate}></Column>
              </DataTable>
            </div>
            <div className="flex mt-4">
              <div className="ml-2 flex flex-row align-items-center justify-content-center">
                <Button
                  label="Zufall"
                  type="button"
                  className="p-button-sm p-button-text m-2"
                  onClick={() => handleRandomDataClick()}
                  tooltip="Zufällig Tweet aus den Testdaten (mixed)"
                  tooltipOptions={{ position: "top" }}
                />
                <h4>Tweet: </h4>
                <InputText
                  className="m-2"
                  style={{ width: "45rem" }}
                  id="tweetInput"
                  value={tweetText}
                  onChange={(e) => setTweetText(e.target.value)}
                  invalid={isInputInvalid()}
                />
                <h4>Label: </h4>
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
                    handleClassificationClickSingle(
                      -1,
                      tweetText,
                      tweetLabel.label,
                    )
                  }
                />
              </div>
            </div>
          </Panel>
        </div>

        <div className="flex flex-column m-2" style={{ width: "46vw" }}>
          <Panel header="Einzelner Tweet" className="mb-3">
            <div className="ml-2 flex flex-row align-items-center">
              <h4>Ausgewählter Tweet: </h4>
              <div className="ml-2">
                {selectedTweet ? selectedTweet["tweet"] : ""}
              </div>
            </div>
            <div className="flex flex-row justify-content-start">
              <div className="ml-2 flex flex-row align-items-center">
                <h4>Label: </h4>
                <div className="ml-2">
                  {selectedTweet ? selectedTweet["label"] : ""}
                </div>
              </div>
            </div>

            <DataTable
              value={predictionData}
              tableStyle={{ width: "45rem", height: "30vh" }}
              scrollable
              scrollHeight="50rem"
              loading={loadingPrediction}
              size="small"
            >
              <Column field="model_name" header="Model Name"></Column>
              <Column
                field="zero_probability"
                header="No Hatespeech-Probability"
              ></Column>
              <Column
                field="one_probability"
                header="Hatespeech-Probability"
              ></Column>
              <Column
                field="label"
                header="Prediction"
                body={predictionTemplate}
              ></Column>
            </DataTable>
          </Panel>
        </div>
      </div>
    </>
  );
};

export default observer(TweetOverviewSingle);
