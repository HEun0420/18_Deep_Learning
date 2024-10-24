from fastapi import FastAPI
from routers import translate


app = FastAPI()


app.include_router(translate.router)
