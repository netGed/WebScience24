import { action, makeAutoObservable, observable } from "mobx";
import {
  TClassificationModelData,
  TTweetData,
  TTweetDataWithClassifications,
} from "../../types.ts";
import {
  getEvaluationData,
  getRandomTestDataMixed,
  getRandomTestDataNew,
  getRandomTestDataOld,
} from "../api/data.ts";
import { getClassificationForTweetAlt } from "../api/classification.ts";

class TweetStore {
  public tweets: TTweetData[] = [];
  public tweetsWithClassifications: TTweetDataWithClassifications[] = [];
  public loading = false;

  constructor() {
    makeAutoObservable(this, {
      _initTweetsWithClassifications: action,
      _updateTweetsWithClassification: action,
      _switchIsLoading: action,
      _setTweets: action,
      loading: observable,
      tweets: observable,
      tweetsWithClassifications: observable,
    });
    void this.loadTweetsTypeEval();
  }

  public async loadTweetsTypeEval() {
    this._switchIsLoading();
    this._resetTweets();

    const newTweets = await getEvaluationData();
    if (newTweets) {
      this._setTweets(newTweets);
      this._initTweetsWithClassifications(newTweets);
    }
    this._switchIsLoading();
  }

  public async loadTweetsTypeOld() {
    this._switchIsLoading();
    this._resetTweets();

    const newTweets = await getRandomTestDataOld();
    if (newTweets) {
      this._setTweets(newTweets);
      this._initTweetsWithClassifications(newTweets);
    }
    this._switchIsLoading();
  }

  public async loadTweetsTypeNew() {
    this._switchIsLoading();
    this._resetTweets();
    const newTweets = await getRandomTestDataNew();
    if (newTweets) {
      this._setTweets(newTweets);
      this._initTweetsWithClassifications(newTweets);
    }
    this._switchIsLoading();
  }

  public async loadTweetsTypeMixed() {
    this._switchIsLoading();
    this._resetTweets();

    const newTweets = await getRandomTestDataMixed();
    if (newTweets) {
      this._setTweets(newTweets);
      this._initTweetsWithClassifications(newTweets);
    }

    this._switchIsLoading();
  }

  public async updateTweetClassifications(
    selectedTweets: TTweetDataWithClassifications[],
  ) {
    this._switchIsLoading();
    const tweetsToClassify: TTweetDataWithClassifications[] = selectedTweets;

    // keine Zeilenauswahl in Tabelle => alle Tweets klassifizieren
    // if (selectedTweets.length === 0) {
    //   tweetsToClassify = this.createTweetsWithDefaultMetrics(this.tweets);
    // }
    // vorerst deaktivieren, da bei 100 tweets extrem langsam / ineffizient programmiert

    for (const tweet of tweetsToClassify) {
      const classification = (await getClassificationForTweetAlt(
        tweet.tweet,
      )) as TClassificationModelData;
      if (classification) {
        tweet.classification_ensemble = classification.classification_ensemble;
        tweet.classification_nb = classification.classification_nb;
        tweet.classification_svm = classification.classification_svm;
        tweet.classification_gru = classification.classification_gru;
        tweet.classification_lstm = classification.classification_lstm;
        tweet.classification_bert = classification.classification_bert;
        tweet.classification_roberta = classification.classification_roberta;
      }
    }

    this._updateTweetsWithClassification(tweetsToClassify);
    this._switchIsLoading();
  }

  private _resetTweets() {
    this.tweets = [];
  }

  private _setTweets(tweets: TTweetData[]) {
    this.tweets = tweets;
  }

  private _switchIsLoading() {
    this.loading = !this.loading;
  }

  private _updateTweetsWithClassification(
    updatedTweets: TTweetDataWithClassifications[],
  ) {
    updatedTweets.forEach((updatedTweet) => {
      this.tweetsWithClassifications.map((oldTweet) =>
        oldTweet.id === updatedTweet.id ? updatedTweet : oldTweet,
      );
    });
  }

  private _initTweetsWithClassifications(
    tweets: TTweetDataWithClassifications[],
  ) {
    this.tweetsWithClassifications =
      this.createTweetsWithDefaultMetrics(tweets);
  }

  private createTweetsWithDefaultMetrics = (tweets: TTweetData[]) => {
    return tweets.map((tweet) => {
      return {
        id: tweet.id,
        tweet: tweet.tweet,
        label: tweet.label,
        classification_ensemble: undefined,
        classification_nb: undefined,
        classification_svm: undefined,
        classification_gru: undefined,
        classification_lstm: undefined,
        classification_bert: undefined,
        classification_roberta: undefined,
      } as TTweetDataWithClassifications;
    });
  };
}

export default new TweetStore();
