import yaml
import networkx as nx
import asyncio
from fastapi import FastAPI, HTTPException
import httpx
import time
from fastapi import Response


app = FastAPI()

MOCK_SERVER_URL = "http://127.0.0.1:8010/long-task"

# Load tasks from YAML
def load_tasks_from_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data["tasks"]

# Build DAG
def build_dag(tasks):
    dag = nx.DiGraph()
    for task_name, task_info in tasks.items():
        dag.add_node(task_name, delay=task_info["delay"])  # Store delay as node attribute
        for dependency in task_info["dependencies"]:
            dag.add_edge(dependency, task_name)
    return dag

# Asynchronous function to simulate a task with a delay
async def call_in_parallel(task_name, delay):
    print(f"+++++++++ Task {task_name}: Delay = {delay} secs")
    response = await fetch_data(url=f"{MOCK_SERVER_URL}?task_name={task_name}&delay={delay}", timeout=20)
    print(f"--------- Task {task_name}: Response: {response}")
    return {"task": task_name, "response": response}

# Execute the DAG
async def execute_dag(dag):
    start_time = time.time()
    task_completion_times = {}

    # topological sorting of the tasks
    sorted_tasks = list(nx.topological_sort(dag))
    completed_tasks = set()

    async def execute_task(task):
        # Wait for dependencies
        for dependency in dag.predecessors(task):
            # wait till the time the dependency is not complete
            while dependency not in completed_tasks:
                await asyncio.sleep(0.1)  # Poll for dependency completion

        # proceed if the dependencies are done / in case of no dependencies

        # Get task delay from DAG attributes
        delay = dag.nodes[task]["delay"]
        # Execute the task
        result = await call_in_parallel(task, delay)
        completed_tasks.add(task)

        task_completion_times[task] = time.time() - start_time

        return result

    # Schedule and execute tasks as their dependencies are resolved
    tasks = [execute_task(task) for task in sorted_tasks]
    results = await asyncio.gather(*tasks)
    timeline = {task: round(task_completion_times[task], 2) for task in sorted_tasks}

    # Sort tasks by completion time and print them in order
    outstr = ""
    for task, completion_time in sorted(task_completion_times.items(), key=lambda x: x[1]):
        outstr += f"Task {task} completed at {round(completion_time, 2)} seconds\n"

    return outstr

# Reuse your fetch_data function
async def fetch_data(url: str, method: str = "GET", payload: dict = None, headers: dict = None, timeout: int = 15) -> dict:
    """
    Fetch data from a given URL using HTTPX with customizable timeout.
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

@app.get("/")
def say_hi():
    return {'message': 'hi robo. Route to /execute_dag to start the DAG execution'}

@app.get("/execute_dag")
async def execute_dag_endpoint():
    yaml_file = "tasks_dag.yaml"
    tasks = load_tasks_from_yaml(yaml_file)
    dag = build_dag(tasks)

    if not nx.is_directed_acyclic_graph(dag):
        raise HTTPException(status_code=400, detail="The task configuration contains cycles!")

    results = await execute_dag(dag)
    return Response(content=results, media_type="text/plain")






















# import yaml

# import networkx as nx
# import asyncio
# from fastapi import FastAPI, HTTPException
# import httpx
# import random

# app = FastAPI()

# MOCK_SERVER_URL = "http://127.0.0.1:8010/long-task?delay="

# # Load tasks from YAML
# def load_tasks_from_yaml(yaml_file):
#     with open(yaml_file, 'r') as f:
#         data = yaml.safe_load(f)
#     return data["tasks"]

# # Build DAG
# def build_dag(tasks):
#     dag = nx.DiGraph()
#     for task_name, task_info in tasks.items():
#         dag.add_node(task_name)
#         for dependency in task_info["dependencies"]:
#             dag.add_edge(dependency, task_name)
#     return dag

# # Asynchronous function to simulate a task
# async def call_in_parallel(task_name, delay):
#     delay = int(delay)
#     # random_delay_secs = random.randint(1, 10)
#     print(f"Task {task_name}: Passed delay = {delay} secs")
#     response = await fetch_data(url=f"{MOCK_SERVER_URL}{delay}", timeout=15)
#     print(f"Task {task_name}: Completed with response: {response}")
#     return {"task": task_name, "response": response}

# # Execute the DAG
# async def execute_dag(dag):
#     sorted_tasks = list(nx.topological_sort(dag))
#     completed_tasks = set()

#     async def execute_task(task):
#         # Wait for dependencies
#         for dependency in dag.predecessors(task):
#             while dependency not in completed_tasks:
#                 await asyncio.sleep(0.1)  # Poll for dependency completion
#         # Execute the task
#         result = await call_in_parallel(task)
#         completed_tasks.add(task)
#         return result

#     # Schedule and execute tasks as their dependencies are resolved
#     tasks = [execute_task(task) for task in sorted_tasks]
#     results = await asyncio.gather(*tasks)
#     return results

# # Reuse your fetch_data function
# async def fetch_data(url: str, method: str = "GET", payload: dict = None, headers: dict = None, timeout: int = 15) -> dict:
#     """
#     Fetch data from a given URL using HTTPX with customizable timeout.
#     """
#     try:
#         async with httpx.AsyncClient(timeout=timeout) as client:
#             if method.upper() == "GET":
#                 response = await client.get(url, headers=headers)
#             elif method.upper() == "POST":
#                 response = await client.post(url, json=payload, headers=headers)
#             elif method.upper() == "PUT":
#                 response = await client.put(url, json=payload, headers=headers)
#             elif method.upper() == "DELETE":
#                 response = await client.delete(url, headers=headers)
#             else:
#                 raise ValueError(f"Unsupported HTTP method: {method}")

#             response.raise_for_status()
#             return response.json()
#     except httpx.RequestError as exc:
#         raise HTTPException(status_code=500, detail=f"Network error while calling {url}: {exc}")
#     except httpx.HTTPStatusError as exc:
#         raise HTTPException(
#             status_code=exc.response.status_code,
#             detail=f"HTTP error while calling {url}: {exc.response.text}"
#         )

# @app.get("/")
# def say_hi():
#     return {'message': 'hi robo. Route to /execute_dag to start the DAG execution'}

# @app.get("/execute_dag")
# async def execute_dag_endpoint():
#     yaml_file = "tasks_dag.yaml"
#     tasks = load_tasks_from_yaml(yaml_file)
#     dag = build_dag(tasks)

#     if not nx.is_directed_acyclic_graph(dag):
#         raise HTTPException(status_code=400, detail="The task configuration contains cycles!")

#     results = await execute_dag(dag)
#     return {"results": results}
