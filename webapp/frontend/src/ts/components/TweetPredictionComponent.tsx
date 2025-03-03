import React from "react";
import {Panel} from "primereact/panel";
import {Column} from "primereact/column";
import {DataTable} from "primereact/datatable";
import {TClassificationData, TTweetData} from "../../types.ts";
import {classNames} from "primereact/utils";
import {observer} from "mobx-react-lite";

type TTweetPredictionComponent = {
    selectedTweet: TTweetData | undefined;
    predictionData: TClassificationData[];
    loadingPrediction: boolean;
}

const TweetPredictionComponent: React.FC<TTweetPredictionComponent> = ({selectedTweet, predictionData, loadingPrediction}) => {
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
            <Panel className="mb-3">

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
                    tableStyle={{width: "45rem", height: "30vh"}}
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
        </>
    );
};

export default observer(TweetPredictionComponent);
