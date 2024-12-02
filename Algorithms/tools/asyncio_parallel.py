from fastapi import FastAPI, HTTPException
import httpx
import random
import asyncio

app = FastAPI()

MOCK_SERVER_URL = "http://127.0.0.1:8010/long-task?delay="



@app.get("/")
def say_hi():
    return {'message': 'hi robo. route to /call_in_parallel for parallelization'}

@app.get("/throw_parallel_calls")
async def throw_parallel_calls():
    # Launch multiple asynchronous calls in parallel
    results = await asyncio.gather(
        call_in_parallel(),
        call_in_parallel(),
        call_in_parallel(),
        call_in_parallel()
    )
    return {"results": results}

async def call_in_parallel():
    random_delay_secs = random.randint(1,10)
    print(f"Passed delay = {random_delay_secs} secs")
    response = await fetch_data(url=f"{MOCK_SERVER_URL}{random_delay_secs}", timeout=11)
    print(f"-------------- Recieved response for {random_delay_secs}")
    return response

async def fetch_data(url: str, method: str = "GET", payload: dict = None, headers: dict = None, timeout: int = 15) -> dict:
    """
    Fetch data from a given URL using HTTPX with customizable timeout.

    Args:
        url (str): The target URL.
        method (str): HTTP method (default: "GET").
        payload (dict): Request payload for POST/PUT requests (default: None).
        headers (dict): Optional HTTP headers (default: None).
        timeout (int): Timeout in seconds (default: 15).

    Returns:
        dict: Parsed JSON response from the server.

    Raises:
        HTTPException: On network or HTTP error.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method.upper() == "GET":
                response = await client.get(url, headers=headers)
            elif method.upper() == "POST":
                response = await client.post(url, json=payload, headers=headers)
            elif method.upper() == "PUT":
                response = await client.put(url, json=payload, headers=headers)
            elif method.upper() == "DELETE":
                response = await client.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=500, detail=f"Network error while calling {url}: {exc}")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"HTTP error while calling {url}: {exc.response.text}"
        )
