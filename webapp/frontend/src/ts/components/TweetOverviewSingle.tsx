import React, {useState} from "react";
import {Panel} from "primereact/panel";
import {Column} from "primereact/column";
import {DataTable, DataTableFilterMeta} from "primereact/datatable";
import {Button} from "primereact/button";
import {TClassificationData, TTweetData} from "../../types.ts";
import {getClassificationForTweet} from "../api/classification.ts";
import {InputText} from "primereact/inputtext";
import {Dropdown, DropdownChangeEvent} from "primereact/dropdown";
import {getRandomTestDataMixed} from "../api/data.ts";
import TweetStore from "../stores/TweetStore.ts";
import {observer} from "mobx-react-lite";
import TweetPredictionComponent from "./TweetPredictionComponent.tsx";
import {FilterMatchMode} from "primereact/api";
import {IconField} from "primereact/iconfield";
import {InputIcon} from "primereact/inputicon";

const labels = [{label: 0}, {label: 1}];

const TweetOverviewSingle: React.FC = () => {
    const [pinnedSelectedTweet, setPinnedSelectedTweet] = useState<TTweetData>();
    const [pinnedPredictionData, setPinnedPredictionData] = useState<TClassificationData[]>(
        [],
    );
    const [loadingPinningPrediction, isLoadingPinningPrediction] = useState(false);

    const [selectedTweet, setSelectedTweet] = useState<TTweetData>();
    const [predictionData, setPredictionData] = useState<TClassificationData[]>(
        [],
    );
    const [loadingPrediction, isLoadingPrediction] = useState(false);
    const [tweetText, setTweetText] = useState<string>("");
    const [tweetLabel, setTweetLabel] = useState(labels[0]);

    const [filters, setFilters] = useState<DataTableFilterMeta>({
        global: {value: null, matchMode: FilterMatchMode.CONTAINS},
        id: {value: null, matchMode: FilterMatchMode.CONTAINS},
        tweet: {value: null, matchMode: FilterMatchMode.CONTAINS},
    });
    const [globalFilterValue, setGlobalFilterValue] = useState<string>('');

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

    const handlePinningClickSingle = async (
        id: number,
        tweet: string,
        label: number,
    ) => {
        isLoadingPinningPrediction(true);
        const tweetData = {
            id: id,
            tweet: tweet,
            label: label,
        };
        setPinnedSelectedTweet(tweetData);

        const predictionResult = (await getClassificationForTweet(
            tweet,
        )) as TClassificationData[];
        setPinnedPredictionData(predictionResult);

        isLoadingPinningPrediction(false);
    };

    const handleRandomDataClick = async () => {
        const data = (await getRandomTestDataMixed()) as TTweetData[];
        if (data.length > 0) {
            setTweetText(data[0].tweet);
            setTweetLabel({label: data[0].label});
        }
    };

    const classificationTemplate = (rowData: TTweetData) => {
        return (
            <div className="flex">
                <Button
                    rounded
                    type="button"
                    icon="pi pi-tags"
                    className="p-button-sm p-button-text"
                    style={{borderBottomRightRadius: "0", borderTopRightRadius: "0"}}
                    severity="info"
                    onClick={() => {
                        handlePinningClickSingle(
                            rowData["id"],
                            rowData["tweet"],
                            rowData["label"],
                        );
                    }}
                />
                <Button
                    rounded
                    type="button"
                    icon="pi pi-tags"
                    className="p-button-sm p-button-text"
                    style={{borderBottomLeftRadius: "0", borderTopLeftRadius: "0"}}
                    severity="success"
                    onClick={() => {
                        handleClassificationClickSingle(
                            rowData["id"],
                            rowData["tweet"],
                            rowData["label"],
                        );
                    }}
                />
            </div>
        );
    };

    const onGlobalFilterChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = e.target.value;
        const _filters = { ...filters };

        // @ts-ignore
        _filters['global'].value = value;

        setFilters(_filters);
        setGlobalFilterValue(value);
    };

    const datatableSearchHeader = () => {
        return (
            <div className="flex justify-content-end">
                <IconField iconPosition="left">
                    <InputIcon className="pi pi-search"/>
                    <InputText value={globalFilterValue} onChange={onGlobalFilterChange} placeholder="Keyword Search"/>
                </IconField>
            </div>
        );
    };

    return (
        <>
            <div className="flex overflow-hidden  ml-3">
                <div className="flex flex-column m-2" style={{width: "50vw"}}>
                    <Panel header="Tweetübersicht">
                        <div className="flex">
                            <DataTable
                                value={TweetStore.tweets}
                                scrollable
                                scrollHeight="50rem"
                                loading={TweetStore.loading}
                                filters={filters}
                                filterDisplay="menu"
                                globalFilterFields={['id', 'tweet']}
                                header={datatableSearchHeader}
                                paginator
                                rows={25}
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
                                    tooltipOptions={{position: "top"}}
                                />
                                <h4>Tweet: </h4>
                                <InputText
                                    className="m-2"
                                    style={{width: "45rem"}}
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

                                <div className="flex">
                                    <Button
                                        rounded
                                        type="button"
                                        icon="pi pi-tags"
                                        className="p-button-sm p-button-text"
                                        style={{borderBottomRightRadius: "0", borderTopRightRadius: "0"}}
                                        severity="info"
                                        onClick={() => {
                                            handlePinningClickSingle(
                                                -1,
                                                tweetText,
                                                tweetLabel.label,
                                            );
                                        }}
                                    />
                                    <Button
                                        rounded
                                        type="button"
                                        icon="pi pi-tags"
                                        className="p-button-sm p-button-text"
                                        style={{borderBottomLeftRadius: "0", borderTopLeftRadius: "0"}}
                                        severity="success"
                                        onClick={() => {
                                            handleClassificationClickSingle(
                                                -1,
                                                tweetText,
                                                tweetLabel.label,
                                            );
                                        }}
                                    />
                                </div>
                            </div>
                        </div>
                    </Panel>
                </div>

                <div className="flex flex-column m-2" style={{width: "46vw"}}>
                    <TweetPredictionComponent selectedTweet={pinnedSelectedTweet} predictionData={pinnedPredictionData} loadingPrediction={loadingPinningPrediction}/>
                    <TweetPredictionComponent selectedTweet={selectedTweet} predictionData={predictionData} loadingPrediction={loadingPrediction}/>
                </div>
            </div>
        </>
    );
};

export default observer(TweetOverviewSingle);
