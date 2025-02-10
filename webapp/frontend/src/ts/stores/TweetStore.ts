import { makeAutoObservable, observable } from "mobx";
import { TTweetData, TTweetDataWithMetric } from "../../types.ts";
import {
  getEvaluationData,
  getRandomTestDataMixed,
  getRandomTestDataNew,
  getRandomTestDataOld,
} from "../api/data.ts";

class TweetStore {
  public tweets: TTweetData[] = [];
  public tweetsWithMetrics: TTweetDataWithMetric[] = [];
  public loading = false;

  constructor() {
    makeAutoObservable(this, {
      tweets: observable,
      tweetsWithMetrics: observable,
      loading: observable,
    });
    void this.loadTweetsTypeEval();
  }

  public async loadTweetsTypeEval() {
    this.loading = true;
    this.tweets = [];

    const newTweets = await getEvaluationData();
    if (newTweets) {
      this.tweets = newTweets;
      this.initializeTweetsWithDefaultMetrics(newTweets);
    }
    this.loading = false;
  }

  public async loadTweetsTypeOld() {
    this.loading = true;
    this.tweets = [];
    const newTweets = await getRandomTestDataOld();
    if (newTweets) {
      this.tweets = newTweets;
      this.initializeTweetsWithDefaultMetrics(newTweets);
    }
    this.loading = false;
  }

  public async loadTweetsTypeNew() {
    this.loading = true;
    this.tweets = [];
    const newTweets = await getRandomTestDataNew();
    if (newTweets) {
      this.tweets = newTweets;
      this.initializeTweetsWithDefaultMetrics(newTweets);
    }
    this.loading = false;
  }

  public async loadTweetsTypeMixed() {
    this.loading = true;
    this.tweets = [];
    const newTweets = await getRandomTestDataMixed();
    if (newTweets) {
      this.tweets = newTweets;
      this.initializeTweetsWithDefaultMetrics(newTweets);
    }
    this.loading = false;
  }

  public async updateTweetMetrics(selectedTweets: TTweetDataWithMetric[]) {
    const tweetToUpdate = [];
    if (selectedTweets.length === 0) {
      tweetToUpdate = this.tweets;
      // todo
    }
  }

  private initializeTweetsWithDefaultMetrics = (tweets: TTweetData[]) => {
    this.tweetsWithMetrics = tweets.map((tweets) => {
      return {
        tweet: tweets.tweet,
        id: tweets.id,
        label: tweets.label,
        ensemble_metric: 0,
        nb_metric: 0,
        svm_metric: 0,
        gru_metric: 0,
        lstm_metric: 0,
        bert_metric: 0,
        roberta_metric: 0,
      } as TTweetDataWithMetric;
    });
  };
}

export default new TweetStore();
