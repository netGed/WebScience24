from pydantic import BaseModel


class TweetData(BaseModel):
    id: int
    tweet: str
    label: int


class Tweet(BaseModel):
    tweet: str
