"""Router do domínio `labs`: handlers das rotas versionadas ``/api/vN`` deste domínio."""

from __future__ import annotations

from fastapi import APIRouter

from ...api_v7 import MAX_BACKTEST_HORIZON
from ...automl.engine import AutoMLConfig, TabularAutoML
from ...config import get_settings
from ...data import TARGET_COLUMN, load_iris_dataframe
from ...errors import invalid_input_as_422
from ...knowledge.graph import lineage_graph
from ...nlp.text_pipeline import TextClassifier
from ...schemas import AutoMLRequest, ForecastRequest, TextRequest
from ...services import get_service
from ...timeseries.baselines import backtest, forecast

router = APIRouter()
settings = get_settings()
service = get_service()


@router.post("/api/v7/labs/automl", tags=["labs"])
def automl(payload: AutoMLRequest) -> dict:
    """Compara por validação cruzada os primeiros `max_models` modelos do catálogo no Iris."""
    config = AutoMLConfig(target=TARGET_COLUMN, max_models=payload.max_models, folds=payload.folds)
    with invalid_input_as_422():
        return TabularAutoML(config).run(load_iris_dataframe())


@router.post("/api/v7/labs/text", tags=["labs"])
def text_lab(payload: TextRequest) -> dict:
    """Treina um classificador TF-IDF + regressão logística com os textos enviados."""
    with invalid_input_as_422():
        model = TextClassifier().fit(payload.texts, payload.labels)
    return {
        "manifest": model.manifest(),
        "predictions": model.predict(payload.query) if payload.query else [],
        "probabilities": model.probabilities(payload.query) if payload.query else [],
    }


@router.post("/api/v7/labs/forecast", tags=["labs"])
def forecast_lab(payload: ForecastRequest) -> dict:
    """Previsão por baseline e backtest de origem deslizante sobre a própria série."""
    values = payload.values
    horizon = min(payload.horizon, MAX_BACKTEST_HORIZON)
    # O backtest exige mais de `min_train + horizon` pontos; com o mínimo aceito
    # pelo schema (8) ainda sobra pelo menos uma janela de avaliação.
    min_train = max(3, min(8, len(values) - horizon - 1))
    with invalid_input_as_422():
        prediction = forecast(values, payload.horizon, payload.method, payload.window)
        evaluation = backtest(values, horizon, payload.method, min_train, payload.window)
    return {
        "forecast": {"method": prediction.method, "horizon": prediction.horizon, "values": prediction.values},
        "backtest": evaluation,
    }


@router.get("/api/v7/labs/lineage-graph", tags=["labs"])
def lineage() -> dict:
    """Grafo dataset -> execução de treino -> versão do modelo -> estágio, lido do registry."""
    return lineage_graph(service.list_models()).manifest()
