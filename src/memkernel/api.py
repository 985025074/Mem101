from fastapi import FastAPI

from memkernel.kernel import PostMemory

app = FastAPI()
# kernel = MemKernel()


@app.get("/")
def home():
    return {"message": "Hello, Memkernel "}


@app.get("/recall/")
def get_item(query: str):

    pass


@app.post("/memory/")
def post_memory(memory: PostMemory):
    pass
