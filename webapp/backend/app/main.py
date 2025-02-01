from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from webapp.backend.app.data_reader import read_data
from webapp.backend.app.predictions import predict_ensemble, predict_svm, predict_nb, predict_gru, \
    predict_lstm, predict_bert, predict_roberta

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
    data = read_data()
    print(data)
    return data


@app.post("/get_predictions")
async def get_predictions(request_body: str = Body(..., media_type="text/plain")):
    classification_results = []

    result_ensemble = predict_ensemble(request_body)
    result_svm = predict_svm(request_body)
    result_nb = predict_nb(request_body)
    result_gru = predict_gru(request_body)
    result_lstm = predict_lstm(request_body)
    result_bert = predict_bert(request_body)
    result_roberta = predict_roberta(request_body)

    classification_results.append(result_ensemble)
    classification_results.append(result_svm)
    classification_results.append(result_nb)
    classification_results.append(result_gru)
    classification_results.append(result_lstm)
    classification_results.append(result_bert)
    classification_results.append(result_roberta)

    return classification_results


@app.post("/get_prediction_ensemble")
async def get_prediction_ensemble(request_body: str = Body(..., media_type="text/plain")):
    classification_result = predict_ensemble(request_body)
    return classification_result


@app.post("/get_prediction_svm")
async def get_prediction_svm(request_body: str = Body(..., media_type="text/plain")):
    classification_result = predict_svm(request_body)
    return classification_result


@app.post("/get_prediction_nb")
async def get_prediction_nb(request_body: str = Body(..., media_type="text/plain")):
    classification_result = predict_nb(request_body)
    return classification_result


@app.post("/get_prediction_lstm")
async def get_prediction_lstm(request_body: str = Body(..., media_type="text/plain")):
    classification_result = predict_lstm(request_body)
    return classification_result


@app.post("/get_prediction_gru")
async def get_prediction_gru(request_body: str = Body(..., media_type="text/plain")):
    classification_result = predict_gru(request_body)
    return classification_result


@app.post("/get_prediction_bert")
async def get_prediction_bert(request_body: str = Body(..., media_type="text/plain")):
    classification_result = predict_bert(request_body)
    return classification_result


@app.post("/get_prediction_roberta")
async def get_prediction_roberta(request_body: str = Body(..., media_type="text/plain")):
    classification_result = predict_roberta(request_body)
    return classification_result


if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()
