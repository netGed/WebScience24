from pydantic import BaseModel


class Tweet(BaseModel):
    id: int
    tweet: str
    label: int
