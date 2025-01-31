import React from "react";
import TweetTable from "./TweetTable.tsx";
import "primeflex/primeflex.css";
import "primereact/resources/themes/lara-light-indigo/theme.css";
import "primeicons/primeicons.css";

const App: React.FC = () => {
  return (
    <>
      <div style={{ width: "75%", height: "75%" }}>
        <TweetTable />
      </div>
    </>
  );
};

export default App;
