from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProductIn(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    name: str = Field(min_length=1)
    manufacturer: str = Field(min_length=1)
    specs: dict[str, str] = Field(default_factory=dict)
    description: str = ""

    @field_validator("specs", mode="before")
    @classmethod
    def _normalize_specs(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError('specs должен быть объектом "характеристика => значение"')
        result: dict[str, str] = {}
        for key, item in value.items():
            if isinstance(item, (dict, list, tuple, set)):
                raise ValueError(
                    f'значение характеристики "{key}" должно быть строкой или числом'
                )
            if item is not None:
                result[str(key).strip()] = str(item).strip()
        return result


class Characteristic(BaseModel):
    name: str
    value: str
    source: Literal["input", "missing"]


class WarningItem(BaseModel):
    code: str
    message: str


class CardDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seo_title: str
    seo_description: str
    search_queries: list[str] = Field(min_length=3, max_length=5)
    commercial_block: str


class GeneratedCard(BaseModel):
    seo_title: str
    seo_description: str
    characteristics: list[Characteristic]
    search_queries: list[str]
    commercial_block: str


class ProductRef(BaseModel):
    name: str
    manufacturer: str


class CardMeta(BaseModel):
    model: str
    latency_ms: int
    attempts: int
    char_counts: dict[str, int]


class CardResponse(BaseModel):
    status: Literal["ok", "completed_with_gaps", "needs_review"]
    product: ProductRef
    card: GeneratedCard
    missing_data: list[str]
    warnings: list[WarningItem]
    meta: CardMeta


class BatchIn(BaseModel):
    products: list[Any] = Field(min_length=1, max_length=50)


class BatchItem(BaseModel):
    index: int
    result: CardResponse | None = None
    error: str | None = None


class BatchMeta(BaseModel):
    processed: int
    succeeded: int
    failed: int


class BatchResponse(BaseModel):
    results: list[BatchItem]
    meta: BatchMeta
