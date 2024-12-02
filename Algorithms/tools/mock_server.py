from fastapi import FastAPI
import asyncio
import random

app = FastAPI()

@app.get("/long-task")
async def long_task(task_name, delay):
    await asyncio.sleep(int(delay))  # Simulates a 10-second task
    return {"message": f"Task {task_name} completed after {delay} seconds"}

@app.get("/")
async def root():
    return {"message": "I am a mock server taking 10 seconds for long-task request and I am alive!"}
