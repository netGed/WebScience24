from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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

@app.post("/gettest")
async def gettest(request_body: str = Body(..., media_type="text/plain")):
    print(request_body)
    classification_result = "99%"
    return {"message": classification_result}



if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()

