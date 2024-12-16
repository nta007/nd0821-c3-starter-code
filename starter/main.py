from fastapi import FastAPI
import uvicorn

app = FastAPI(version="0.1.0", title="Udacity DevOps ML API",
              description="API for ML model")

if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=True)


@app.route('/')
def home():
    return {"message": "Hello World!"}
