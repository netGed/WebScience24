from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from webapp.backend.app.classification_metrics import generate_classification_metrics_for_ensemble, \
    generate_classification_metrics_for_svm, generate_classification_metrics_for_nb, \
    generate_classification_metrics_for_gru, generate_classification_metrics_for_lstm, \
    generate_classification_metrics_for_bert, generate_classification_metrics_for_roberta
from webapp.backend.app.data_reader import get_data_evaluation, get_data_mixed, get_data_new, \
    get_data_old
from webapp.backend.app.classifications import classify_with_ensemble, classify_with_svm, classify_with_nb, \
    classify_with_gru, \
    classify_with_lstm, classify_with_bert, classify_with_roberta
from webapp.backend.app.types import Tweet

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "content"}


@app.get("/get_evaluation_data")
async def get_evaluation_data():
    data = get_data_evaluation()
    return data


@app.post("/get_random_test_data_old")
async def get_random_test_data_old(count: int = 1):
    data = get_data_old(count)
    return data


@app.post("/get_random_test_data_new")
async def get_random_test_data_new(count: int = 1):
    data = get_data_new(count)
    return data


@app.post("/get_random_test_data_mixed")
async def get_random_test_data_mixed(count: int = 1):
    data = get_data_mixed(count)
    return data


@app.post("/get_classification_data")
async def get_classification_data(request_body: str = Body(..., media_type="text/plain")):
    classification_data = []

    result_ensemble = classify_with_ensemble(request_body)
    result_svm = classify_with_svm(request_body)
    result_nb = classify_with_nb(request_body)
    result_gru = classify_with_gru(request_body)
    result_lstm = classify_with_lstm(request_body)
    result_bert = classify_with_bert(request_body)
    result_roberta = classify_with_roberta(request_body)

    classification_data.append(result_ensemble)
    classification_data.append(result_svm)
    classification_data.append(result_nb)
    classification_data.append(result_gru)
    classification_data.append(result_lstm)
    classification_data.append(result_bert)
    classification_data.append(result_roberta)

    return classification_data


@app.post("/get_classification_data_alt")
async def get_classification_data_alt(request_body: str = Body(..., media_type="text/plain")):
    result_ensemble = classify_with_ensemble(request_body)
    result_svm = classify_with_svm(request_body)
    result_nb = classify_with_nb(request_body)
    result_gru = classify_with_gru(request_body)
    result_lstm = classify_with_lstm(request_body)
    result_bert = classify_with_bert(request_body)
    result_roberta = classify_with_roberta(request_body)

    result = {
        "classification_ensemble" : result_ensemble["label"],
        "classification_svm": result_svm["label"],
        "classification_nb": result_nb["label"],
        "classification_gru": result_gru["label"],
        "classification_lstm": result_lstm["label"],
        "classification_bert": result_bert["label"],
        "classification_roberta": result_roberta["label"],
    }
    return result


@app.post("/get_classification_metrics")
async def get_classification_metrics(tweets: list[Tweet]):
    classification_metrics = []

    result_ensemble = generate_classification_metrics_for_ensemble(tweets)
    result_svm = generate_classification_metrics_for_svm(tweets)
    result_nb = generate_classification_metrics_for_nb(tweets)
    result_gru = generate_classification_metrics_for_gru(tweets)
    result_lstm = generate_classification_metrics_for_lstm(tweets)
    result_bert = generate_classification_metrics_for_bert(tweets)
    result_roberta = generate_classification_metrics_for_roberta(tweets)

    classification_metrics.append(result_ensemble)
    classification_metrics.append(result_svm)
    classification_metrics.append(result_nb)
    classification_metrics.append(result_gru)
    classification_metrics.append(result_lstm)
    classification_metrics.append(result_bert)
    classification_metrics.append(result_roberta)

    return classification_metrics

@app.post("/get_prediction_ensemble")
async def get_prediction_ensemble(request_body: str = Body(..., media_type="text/plain")):
    classification_result = classify_with_ensemble(request_body)
    return classification_result


@app.post("/get_prediction_svm")
async def get_prediction_svm(request_body: str = Body(..., media_type="text/plain")):
    classification_result = classify_with_svm(request_body)
    return classification_result


@app.post("/get_prediction_nb")
async def get_prediction_nb(request_body: str = Body(..., media_type="text/plain")):
    classification_result = classify_with_nb(request_body)
    return classification_result


@app.post("/get_prediction_lstm")
async def get_prediction_lstm(request_body: str = Body(..., media_type="text/plain")):
    classification_result = classify_with_lstm(request_body)
    return classification_result


@app.post("/get_prediction_gru")
async def get_prediction_gru(request_body: str = Body(..., media_type="text/plain")):
    classification_result = classify_with_gru(request_body)
    return classification_result


@app.post("/get_prediction_bert")
async def get_prediction_bert(request_body: str = Body(..., media_type="text/plain")):
    classification_result = classify_with_bert(request_body)
    return classification_result


@app.post("/get_prediction_roberta")
async def get_prediction_roberta(request_body: str = Body(..., media_type="text/plain")):
    classification_result = classify_with_roberta(request_body)
    return classification_result


if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()
