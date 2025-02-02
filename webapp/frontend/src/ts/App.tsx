import React, { useEffect, useState } from "react";
import "primeflex/primeflex.css";
import "primereact/resources/themes/lara-light-indigo/theme.css";
import "primeicons/primeicons.css";
import { Panel } from "primereact/panel";
import { Column } from "primereact/column";
import { DataTable } from "primereact/datatable";
import { Button } from "primereact/button";
import { TClassificationData, TPredictionData, TTweetData } from "../types.ts";
import {
  getClassificationResults,
  getPredictions,
} from "../api/predictions.ts";
import { classNames } from "primereact/utils";
import { InputText } from "primereact/inputtext";
import { Dropdown, DropdownChangeEvent } from "primereact/dropdown";
import { getEvaluationData, getRandomTestData } from "../api/data.ts";
import Plot from "react-plotly.js";

const labels = [{ label: 0 }, { label: 1 }];
const metrics = [
  { metric: "f1_score" },
  { metric: "accuracy" },
  { metric: "precision" },
  { metric: "recall" },
];

type TPlotData = {
  x: string[];
  y: number[];
};

const App: React.FC = () => {
  const [tweetData, setTweetData] = useState<TTweetData[]>([]);
  const [selectedTweet, setSelectedTweet] = useState<TTweetData>();
  const [selectedTweets, setSelectedTweets] = useState<TTweetData[]>([]);
  const [predictionData, setPredictionData] = useState<TPredictionData[]>([]);
  const [classificationData, setClassificationData] = useState<
    TClassificationData[]
  >([]);
  useState<TTweetData>();
  const [loadingPrediction, isLoadingPrediction] = useState(false);
  const [loadingClassification, isLoadingClassification] = useState(false);
  const [tweetText, setTweetText] = useState<string>("");
  const [tweetLabel, setTweetLabel] = useState(labels[0]);
  const [plotData, setPlotData] = useState<TPlotData>({ x: [], y: [] });
  const [selectedMetric, setSelectedMetric] = useState(metrics[0]);

  const isInputInvalid = () => {
    return tweetText.length < 1;
  };

  useEffect(() => {
    const loadDataAsync = async () => {
      const data = await getEvaluationData();
      setTweetData(data as TTweetData[]);
    };

    loadDataAsync();
  }, []);

  const updatePlotData = (newData: TClassificationData[]) => {
    const x: string[] = [];
    const y: number[] = [];
    const key = selectedMetric.metric as keyof TClassificationData;

    newData.forEach((classData) => {
      x.push(classData.model_name);
      y.push(classData[key] as number);
    });
    setPlotData({ x, y } as TPlotData);
  };

  const handleClassificationClickSingle = async (
    tweet: string,
    label: number,
    new_label: number,
  ) => {
    isLoadingPrediction(true);
    isLoadingClassification(true);
    const tweetData = {
      id: -1,
      tweet: tweet,
      label: label,
      new_label: new_label,
    };
    setSelectedTweet(tweetData);

    const predictionResult = (await getPredictions(tweet)) as TPredictionData[];
    setPredictionData(predictionResult);

    const result = (await getClassificationResults([
      tweetData!,
    ])) as TClassificationData[];
    setClassificationData(result);

    isLoadingPrediction(false);
    isLoadingClassification(false);
  };

  const handleClassificationClickMultiple = async () => {
    isLoadingClassification(true);
    const request = selectedTweets.length > 0 ? selectedTweets : tweetData;
    const result = (await getClassificationResults(
      request,
    )) as TClassificationData[];
    setClassificationData(result);
    isLoadingClassification(false);

    updatePlotData(result);
  };

  const handleRandomDataClick = async () => {
    const data = (await getRandomTestData()) as TTweetData[];
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
            rowData["tweet"],
            rowData["label"],
            rowData["new_label"],
          );
        }}
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

  const newPredictionTemplate = (rowData: TPredictionData) => {
    if (selectedTweet) {
      const realLabel = selectedTweet.new_label;

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

  const plotHeaderTemplate = () => {
    return (
      <Dropdown
        value={selectedMetric}
        onChange={(e: DropdownChangeEvent) => {
          setSelectedMetric(e.value);
          updatePlotData(classificationData);
        }}
        options={metrics}
        optionLabel="metric"
        defaultValue={"f1_score"}
        className="w-full md:w-10rem m-2"
      />
    );
  };

  const classifyAllFooterButton = () => {
    return (
      <div className="flex flex-wrap align-items-center justify-content-end gap-2">
        <Button
          label={selectedTweets.length > 0 ? "Ausgewählte Tweets" : "Alle"}
          className="p-button-text"
          icon="pi pi-tags"
          rounded
          raised
          onClick={() => handleClassificationClickMultiple()}
        />
      </div>
    );
  };

  return (
    <>
      <Panel header="Praktikum WebScience">
        <div className="flex">
          <div className="flex flex-column m-2" style={{ width: "55%" }}>
            <Panel header="Tweetübersicht">
              <div className="flex">
                <DataTable
                  value={tweetData}
                  tableStyle={{ minWidth: "50rem" }}
                  scrollable
                  scrollHeight="50rem"
                  footer={classifyAllFooterButton}
                  selectionMode="multiple"
                  selection={selectedTweets}
                  onSelectionChange={(e) =>
                    setSelectedTweets(e.value as TTweetData[])
                  }
                >
                  <Column field="id" header="Id"></Column>
                  <Column field="tweet" header="Tweet"></Column>
                  <Column field="label" header="Label"></Column>
                  <Column field="new_label" header="Label (neu)"></Column>
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
                    onChange={(e: DropdownChangeEvent) =>
                      setTweetLabel(e.value)
                    }
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
                        tweetText,
                        tweetLabel.label,
                        tweetLabel.label,
                      )
                    }
                  />
                </div>
              </div>
            </Panel>
          </div>

          <div className="flex flex-column m-2" style={{ width: "45%" }}>
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
                <div className="ml-2 flex flex-row align-items-center">
                  <h4>Label (neu): </h4>
                  <div className="ml-2">
                    {selectedTweet ? selectedTweet["new_label"] : ""}
                  </div>
                </div>
              </div>

              <DataTable
                value={predictionData}
                tableStyle={{ minWidth: "45rem" }}
                scrollable
                scrollHeight="50rem"
                loading={loadingPrediction}
                size="small"
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
                <Column
                  field="new_label"
                  header="Prediction (neu)"
                  body={newPredictionTemplate}
                ></Column>
              </DataTable>
            </Panel>

            <Panel header="Klassifikationsmetriken">
              <DataTable
                value={classificationData}
                tableStyle={{ minWidth: "50rem" }}
                scrollable
                scrollHeight="50rem"
                loading={loadingClassification}
                size="small"
              >
                <Column field="model_name" header="Model Name"></Column>
                <Column field="accuracy" header="Accuracy"></Column>
                <Column field="f1_score" header="F1-Score"></Column>
                <Column field="precision" header="Precision"></Column>
                <Column field="recall" header="Recall"></Column>
              </DataTable>
            </Panel>
            <Panel header={plotHeaderTemplate()} className="mt-3">
              <div className="flex justify-content-center">
                <Plot
                  data={[{ type: "bar", x: plotData.x, y: plotData.y }]}
                  layout={{
                    width: 900,
                    height: 650,
                    // title: { text: "Plot" },
                    yaxis: { autorange: false, range: [0, 1] },
                  }}
                />
              </div>
            </Panel>
          </div>
        </div>
      </Panel>
    </>
  );
};

export default App;
