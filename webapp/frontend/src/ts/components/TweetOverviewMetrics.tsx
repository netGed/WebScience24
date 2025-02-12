import React, { useState } from "react";
import { Panel } from "primereact/panel";
import { Column } from "primereact/column";
import { DataTable } from "primereact/datatable";
import { Button } from "primereact/button";
import { TClassificationDataWithMetrics, TTweetData } from "../../types.ts";
import { getClassificationMetrics } from "../api/classification.ts";
import { Dropdown, DropdownChangeEvent } from "primereact/dropdown";
import Plot from "react-plotly.js";
import TweetStore from "../stores/TweetStore.ts";
import { observer } from "mobx-react-lite";

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

const TweetOverviewMetrics: React.FC = () => {
  const [selectedTweets, setSelectedTweets] = useState<TTweetData[]>([]);
  const [classificationData, setClassificationData] = useState<
    TClassificationDataWithMetrics[]
  >([]);
  useState<TTweetData>();
  const [loadingClassification, isLoadingClassification] = useState(false);
  const [plotData, setPlotData] = useState<TPlotData>({ x: [], y: [] });
  const [selectedMetric, setSelectedMetric] = useState(metrics[0]);

  const updatePlotData = (newData: TClassificationDataWithMetrics[]) => {
    const x: string[] = [];
    const y: number[] = [];
    const key = selectedMetric.metric as keyof TClassificationDataWithMetrics;

    newData.forEach((classData) => {
      x.push(classData.model_name);
      y.push(classData[key] as number);
    });
    setPlotData({ x, y } as TPlotData);
  };

  const handleClassificationClickMultiple = async () => {
    isLoadingClassification(true);
    const request =
      selectedTweets.length > 0 ? selectedTweets : TweetStore.tweets;
    const result = (await getClassificationMetrics(
      request,
    )) as TClassificationDataWithMetrics[];
    setClassificationData(result);
    isLoadingClassification(false);

    updatePlotData(result);
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
      <div className="flex overflow-hidden" style={{ height: "85vh" }}>
        <div className="flex flex-column m-2" style={{ width: "50vw" }}>
          <Panel header="Tweetübersicht">
            <div className="flex">
              <DataTable
                value={TweetStore.tweets}
                scrollable
                scrollHeight="50rem"
                footer={classifyAllFooterButton}
                selectionMode="multiple"
                selection={selectedTweets}
                onSelectionChange={(e) =>
                  setSelectedTweets(e.value as TTweetData[])
                }
                loading={TweetStore.loading}
              >
                <Column field="id" header="Id"></Column>
                <Column field="tweet" header="Tweet"></Column>
                <Column field="label" header="Label"></Column>
                <Column field="classify" header=""></Column>
              </DataTable>
            </div>
          </Panel>
        </div>

        <div className="flex flex-column m-2" style={{ width: "46vw" }}>
          <Panel header="Klassifikationsmetriken">
            <DataTable
              value={classificationData}
              tableStyle={{ width: "50rem", height: "30vh" }}
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
                  height: 400,
                  yaxis: { autorange: false, range: [0, 1] },
                }}
              />
            </div>
          </Panel>
        </div>
      </div>
    </>
  );
};

export default observer(TweetOverviewMetrics);
