import React, { useState } from "react";
import { Panel } from "primereact/panel";
import { Column } from "primereact/column";
import { DataTable } from "primereact/datatable";
import { Button } from "primereact/button";
import { TTweetDataWithClassifications } from "../../types.ts";
import TweetStore from "../stores/TweetStore.ts";
import { observer } from "mobx-react-lite";

const TweetOverviewMultiple: React.FC = () => {
  const [selectedTweets, setSelectedTweets] = useState<
    TTweetDataWithClassifications[]
  >([]);

  const classifyAllFooterButton = () => {
    return (
      <div className="flex flex-wrap align-items-center justify-content-end gap-2">
        <Button
          label={selectedTweets.length > 0 ? "Ausgewählte Tweets" : "Alle"}
          className="p-button-text"
          icon="pi pi-tags"
          rounded
          raised
          onClick={() => TweetStore.updateTweetClassifications(selectedTweets)}
        />
      </div>
    );
  };

  return (
    <>
      <div className="flex">
        <div
          className="flex flex-column m-2"
          style={{ height: "85vh", width: "97vw" }}
        >
          <Panel header="Tweetübersicht">
            <div className="flex">
              <DataTable
                value={TweetStore.tweetsWithClassifications}
                scrollable
                scrollHeight="45rem"
                footer={classifyAllFooterButton}
                selectionMode="multiple"
                selection={selectedTweets}
                onSelectionChange={(e) =>
                  setSelectedTweets(e.value as TTweetDataWithClassifications[])
                }
                loading={TweetStore.loading}
              >
                <Column field="id" header="Id"></Column>
                <Column field="tweet" header="Tweet"></Column>
                <Column field="label" header="Label"></Column>
                <Column
                  field="classification_ensemble"
                  header="Ensemble"
                ></Column>
                <Column field="classification_nb" header="Naive Bayes"></Column>
                <Column field="classification_svm" header="SVM"></Column>
                <Column field="classification_gru" header="RNN-GRU"></Column>
                <Column field="classification_lstm" header="RNN-LSTM"></Column>
                <Column field="classification_bert" header="BERT"></Column>
                <Column
                  field="classification_roberta"
                  header="RoBERTa"
                ></Column>
              </DataTable>
            </div>
          </Panel>
        </div>
      </div>
    </>
  );
};

export default observer(TweetOverviewMultiple);
