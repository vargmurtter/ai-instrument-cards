import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from app.config import MODEL, OLLAMA_HOST
from app.schemas import (
    BatchIn,
    BatchItem,
    BatchMeta,
    BatchResponse,
    CardResponse,
    ProductIn,
)
from app.service import generate_card, ollama_status

app = FastAPI()


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "ollama": {"host": OLLAMA_HOST, "model": MODEL, **(await ollama_status())},
    }


@app.post("/api/cards", response_model=CardResponse)
async def create_card(product: ProductIn) -> CardResponse:
    return await generate_card(product)


@app.post("/api/cards/batch", response_model=BatchResponse)
async def create_cards(payload: BatchIn) -> BatchResponse:
    results: list[BatchItem] = []
    for index, raw in enumerate(payload.products):
        try:
            product = ProductIn.model_validate(raw)
        except ValidationError as exc:
            message = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            )
            results.append(BatchItem(index=index, error=message))
            continue
        try:
            results.append(BatchItem(index=index, result=await generate_card(product)))
        except HTTPException as exc:
            results.append(BatchItem(index=index, error=str(exc.detail)))

    failed = sum(1 for item in results if item.error)
    return BatchResponse(
        results=results,
        meta=BatchMeta(
            processed=len(results), succeeded=len(results) - failed, failed=failed
        ),
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
