import React, {useState} from "react";
import "primeflex/primeflex.css";
import "primereact/resources/themes/md-light-indigo/theme.css";
import "primeicons/primeicons.css";
import TweetOverviewSingle from "./components/TweetOverviewSingle.tsx";
import {Button} from "primereact/button";
import TweetStore from "./stores/TweetStore.ts";
import {Tooltip} from "react-tooltip";

const App: React.FC = () => {
    const [isButton1Active, setIsButton1Active] = useState<boolean>(false);
    const [isButton2Active, setIsButton2Active] = useState<boolean>(false);
    const [isButton3Active, setIsButton3Active] = useState<boolean>(false);
    const [isButton4Active, setIsButton4Active] = useState<boolean>(true);

    const setAllButtonsInactive = () => {
        setIsButton1Active(false);
        setIsButton2Active(false);
        setIsButton3Active(false);
        setIsButton4Active(false);
    }

    return (
        <>
            <div className="flex flex-column">
                <div className="flex flex-row justify-content-between">
                    <div
                        className="flex flex-row m-3"
                        style={{height: "5vh"}}
                    >
                        <Button
                            className="p-button-text"
                            style={{borderBottomRightRadius: "0", borderTopRightRadius: "0"}}
                            label="Tweets laden (old)"
                            onClick={() => {
                                TweetStore.loadTweetsTypeOld();
                                setAllButtonsInactive();
                                setIsButton1Active(true);
                            }}
                            severity={isButton1Active ? 'success' : 'info'}
                            outlined
                        />
                        <Button
                            className="p-button-text"
                            style={{borderRadius: "0"}}
                            label="Tweets laden (new)"
                            onClick={() => {
                                TweetStore.loadTweetsTypeNew();
                                setAllButtonsInactive();
                                setIsButton2Active(true);
                            }}
                            severity={isButton2Active ? 'success' : 'info'}
                            outlined
                        />
                        <Button
                            className="p-button-text"
                            style={{borderRadius: "0"}}
                            label="Tweets laden (mixed)"
                            onClick={() => {
                                TweetStore.loadTweetsTypeMixed();
                                setAllButtonsInactive();
                                setIsButton3Active(true);
                            }}
                            severity={isButton3Active ? 'success' : 'info'}
                            outlined
                        />
                        <Button
                            className="p-button-text"
                            style={{borderBottomLeftRadius: "0", borderTopLeftRadius: "0"}}
                            label="Tweets laden (eval)"
                            onClick={() => {
                                TweetStore.loadTweetsTypeEval();
                                setAllButtonsInactive();
                                setIsButton4Active(true);
                            }}
                            severity={isButton4Active ? 'success' : 'info'}
                            outlined
                        />
                    </div>

                    <div className="flex m-3"
                    >
                        <Button
                            label="Ensemble"
                            outlined
                            className="ensembleAnchor"
                        />
                        <Tooltip anchorSelect=".ensembleAnchor" place="bottom">
                            <h4>Modell: BalancedRandomForest</h4>
                            <h4>Vektorisierung: TF-IDF</h4>
                            <h4>Parameter: default</h4>
                        </Tooltip>
                        <Button
                            label="SVM"
                            outlined
                            className="svcAnchor"
                        />
                        <Tooltip anchorSelect=".svcAnchor" place="bottom">
                            <h4>Modell: Support Vector Machine</h4>
                            <h4>Vektorisierung: TF-IDF</h4>
                            <h4>Kernel: linear </h4>
                        </Tooltip>
                        <Button
                            label="Naive Bayes"
                            outlined
                            className="bayesAnchor"
                        />
                        <Tooltip anchorSelect=".bayesAnchor" place="bottom">
                            <h4>Modell: ComplementNB</h4>
                            <h4>Vektorisierung: TF-IDF</h4>
                            <h4>Parameter: </h4>
                        </Tooltip>
                        <Button
                            label="RNN-GRU"
                            outlined
                            className="gruAnchor"
                        />
                        <Tooltip anchorSelect=".gruAnchor" place="bottom">
                            <h4>Modell: RNN-GRU</h4>
                            <h4>Tokenisierung: keras-Tokenizer</h4>
                            <h4>Layer: Embedding, SpatialDropout,GRU,Dense</h4>
                        </Tooltip>
                        <Button
                            label="RNN-LSTM"
                            outlined
                            className="lstmAnchor"
                        />
                        <Tooltip anchorSelect=".lstmAnchor" place="bottom">
                            <h4>Modell: RNN-LSTM</h4>
                            <h4>Tokenisierung: keras-Tokenizer</h4>
                            <h4>Vektorisierung: GloVe</h4>
                            <h4>Layer: Embedding,LSTM,Dropout,Dense</h4>
                        </Tooltip>
                        <Button
                            label="BERT"
                            outlined
                            className="bertAnchor"
                        />
                        <Tooltip anchorSelect=".bertAnchor" place="bottom">
                            <h4>Modell: bert-based-uncased (nachtrainiert)</h4>
                            <h4>Training: 5 Epochen, F1-Scoring</h4>
                        </Tooltip>
                        <Button
                            label="RoBERTa"
                            outlined
                            className="robertaAnchor"
                        />
                        <Tooltip anchorSelect=".robertaAnchor" place="bottom">
                            <h4>Modell: roberta-hate-base (nachtrainiert)</h4>
                            <h4>Training: 5 Epochen, F1-Scoring</h4>
                        </Tooltip>
                    </div>
                </div>
                <div style={{height: "95vh"}}>
                    {/*<TabView>*/}
                    {/*<TabPanel header="Kompaktansicht">*/}
                    {/*  <TweetOverviewCombined />*/}
                    {/*</TabPanel>*/}
                    {/*<TabPanel header="Einzel-Tweet-Vorhersage">*/}
                    <TweetOverviewSingle/>
                    {/*</TabPanel>*/}
                    {/*<TabPanel header="Metriken / Plot">*/}
                    {/*  <TweetOverviewMultiple />*/}
                    {/*</TabPanel>*/}
                    {/*<TabPanel header="Modellvergleich">*/}
                    {/*  <TweetOverviewMultipleNew />*/}
                    {/*</TabPanel>*/}
                    {/*</TabView>*/}
                </div>
            </div>
        </>
    )
        ;
};

export default App;
