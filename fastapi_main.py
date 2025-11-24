# Tutorial: https://fastapi.tiangolo.com/tutorial/
# Doc: http://127.0.0.1:8000/docs or http://127.0.0.1:8000/redoc
from fastapi import FastAPI
import argparse
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
    
    parser = argparse.ArgumentParser(description="Run the FastAPI app.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the app on")
    parser.add_argument("--port", type=int, default=443, help="Port to run the app on")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)