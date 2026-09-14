from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.health import router as health_router
from .api.experiments import router as experiments_router
from .api.model import router as model_router
from .api.prediction import router as prediction_router

app = FastAPI(
    title="EFTA Healthcare XAI API",
    version="1.0.0",
    description="Evaluation Before Trust: non-compensatory safety gates under distribution shift.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(experiments_router)
app.include_router(model_router)
app.include_router(prediction_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "EFTA Healthcare XAI API", "docs": "/docs"}
