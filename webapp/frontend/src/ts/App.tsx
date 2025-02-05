import React from "react";
import "primeflex/primeflex.css";
import "primereact/resources/themes/md-light-indigo/theme.css";
import "primeicons/primeicons.css";
import TweetOverviewSingle from "./components/TweetOverviewSingle.tsx";
import { TabPanel, TabView } from "primereact/tabview";
import TweetOverviewMultiple from "./components/TweetOverviewMetrics.tsx";
import TweetOverviewCombined from "./components/TweetOverviewCombined.tsx";
import TweetOverviewMultipleNew from "./components/TweetOverviewMultiple.tsx";
import { Button } from "primereact/button";
import TweetStore from "./stores/TweetStore.ts";

const App: React.FC = () => {
  return (
    <>
      <div className="flex flex-column">
        <div
          className="flex flex-row justify-content-end m-3"
          style={{ height: "5vh" }}
        >
          <Button
            className="p-button-text"
            style={{ borderBottomRightRadius: "0", borderTopRightRadius: "0" }}
            label="Tweets laden (old)"
            onClick={() => TweetStore.loadTweetsTypeOld()}
          />
          <Button
            className="p-button-text"
            style={{ borderRadius: "0" }}
            label="Tweets laden (new)"
            onClick={() => TweetStore.loadTweetsTypeNew()}
          />
          <Button
            className="p-button-text"
            style={{ borderRadius: "0" }}
            label="Tweets laden (mixed)"
            onClick={() => TweetStore.loadTweetsTypeMixed()}
          />
          <Button
            className="p-button-text"
            style={{ borderBottomLeftRadius: "0", borderTopLeftRadius: "0" }}
            label="Tweets laden (eval)"
            onClick={() => TweetStore.loadTweetsTypeEval()}
          />
        </div>
        <div style={{ height: "95vh" }}>
          <TabView>
            <TabPanel header="Komplettansicht">
              <TweetOverviewCombined />
            </TabPanel>
            <TabPanel header="Einzel-Tweet-Vorhersage">
              <TweetOverviewSingle />
            </TabPanel>
            <TabPanel header="Metriken / Plot">
              <TweetOverviewMultiple />
            </TabPanel>
            <TabPanel header="Modellvergleich">
              <TweetOverviewMultipleNew />
            </TabPanel>
          </TabView>
        </div>
      </div>
    </>
  );
};

export default App;
