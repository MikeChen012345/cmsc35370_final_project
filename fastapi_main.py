# Tutorial: https://fastapi.tiangolo.com/tutorial/
# Doc: http://127.0.0.1:8000/docs or http://127.0.0.1:8000/redoc
from fastapi import FastAPI
import asyncio
from stub import result

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.get("/get_result")
async def get_result(query: str):
    """
    Process the query and return the result.
    """
    return {"result": result(query)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)