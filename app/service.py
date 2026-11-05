import json
import re
import time
from typing import Any

import httpx
import ollama
from fastapi import HTTPException

from app.config import MODEL, OLLAMA_HOST, TIMEOUT
from app.schemas import (
    CardDraft,
    CardMeta,
    CardResponse,
    Characteristic,
    GeneratedCard,
    ProductIn,
    ProductRef,
    WarningItem,
)

MAX_ATTEMPTS = 2

EXPECTED_SPECS = (
    "диапазон измерений",
    "класс точности",
    "присоединение",
    "диаметр корпуса",
    "тип корпуса",
    "питание",
    "интерфейсы",
)
MISSING_VALUE = "данных нет"
TITLE_MAX = 70
DESCRIPTION_MIN = 800
DESCRIPTION_MAX = 1200
QUERIES_MIN = 3
QUERIES_MAX = 5
DESCRIPTION_LIMIT = 6000

SYSTEM_PROMPT = """Ты — редактор B2B-каталога контрольно-измерительных приборов (КИП). \
По данным о приборе ты пишешь черновик карточки товара для интернет-магазина. Пишешь по-русски, \
деловым языком, без воды, эмодзи, markdown и рекламных обещаний.

ЖЁСТКИЕ ПРАВИЛА:
1. Единственный источник фактов — блок «Данные товара». Не добавляй ничего от себя: ни характеристики, \
ни числа, ни единицы измерения, ни сертификаты, ни гарантию, ни сферы применения, ни преимущества, \
которых нет в данных.
2. Не придумывай значения отсутствующих параметров и не заменяй их догадкой. Характеристики, \
отмеченные как отсутствующие, в тексте не упоминай вообще. Чаще всего модель выдумывает диапазон \
рабочих температур, степень защиты IP, гарантийный срок, срок службы, сертификаты и материалы \
уплотнений: если этих сведений нет во входных данных, писать их нельзя.
3. Любое число и единица измерения в твоём тексте должны встречаться во входных данных в том же виде \
(например, «0–1,6 МПа» можно, «до 2,5 МПа» нельзя, если такого значения нет).
4. Не используй словосочетание «данных нет» в тексте карточки — просто не упоминай этот параметр.
5. Оценивай только то, что подтверждается данными. Если сведений мало, пиши короче и нейтральнее, \
но не выходи за нижнюю границу объёма описания.

ТРЕБОВАНИЯ К ПОЛЯМ:
- seo_title: 50–70 символов, включая название прибора и 1–2 ключевых параметра из данных. \
Без точки в конце, без КАПСА, без кавычек, без названия магазина.
- seo_description: 900–1150 символов, жёсткий минимум — 800. Ровно 3 абзаца, в каждом 4–5 полных \
предложений:
  1) что это за прибор, для чего предназначен, измеряемая среда, диапазон и класс точности из данных;
  2) конструкция и условия эксплуатации: корпус, присоединение, монтаж, обслуживание — только из данных;
  3) какие задачи решает прибор и кому подходит: инженерам, снабженцам, монтажным организациям.
  Не заканчивай текст, пока не наберётся 900 символов. Описание короче 800 символов — грубая ошибка.
- search_queries: 3–5 поисковых фраз, которые реально вводит инженер или снабженец. Формулировки \
естественные, на русском, без кавычек, без названия магазина; бренд допустим не более чем в одной фразе.
- commercial_block: 2 абзаца, 350–800 символов. Кому подходит прибор и какие задачи решает. \
Без призывов «купить сейчас» и без обещаний, которых нет в данных.

Ответ — только JSON-объект по заданной схеме, без пояснений и без markdown."""

CARD_SCHEMA: dict[str, Any] = CardDraft.model_json_schema()

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_SAFE_NUMBERS = {str(value) for value in range(10)}


def _norm(key: str) -> str:
    return re.sub(r"\s+", " ", key.replace("ё", "е").replace("Ё", "Е").strip().lower())


def prepare(product: ProductIn) -> tuple[list[tuple[str, str]], list[str]]:
    by_key: dict[str, tuple[str, str]] = {}
    for raw_key, raw_value in product.specs.items():
        key = _norm(raw_key)
        if key and key not in by_key:
            by_key[key] = (raw_key.strip(), str(raw_value).strip())

    specs: list[tuple[str, str]] = []
    missing: list[str] = []
    for canonical in EXPECTED_SPECS:
        entry = by_key.pop(canonical, None)
        if entry is None:
            missing.append(canonical)
        else:
            specs.append(entry)
    specs.extend(by_key[key] for key in sorted(by_key))
    return specs, missing


def characteristics(specs: list[tuple[str, str]]) -> list[Characteristic]:
    # Характеристики собираются кодом из входных данных, чтобы модель не могла их выдумать.
    present = {_norm(name): value for name, value in specs}
    canonical = set(EXPECTED_SPECS)
    result = [
        Characteristic(
            name=name,
            value=present.get(name, MISSING_VALUE),
            source="input" if name in present else "missing",
        )
        for name in EXPECTED_SPECS
    ]
    result.extend(
        Characteristic(name=name, value=value, source="input")
        for name, value in specs
        if _norm(name) not in canonical
    )
    return result


def _number(token: str) -> str:
    value = token.replace(",", ".")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    if value.startswith("0") and len(value) > 1 and value[1] != ".":
        value = value.lstrip("0") or "0"
    return value


def unverified_numbers(prose: str, source: str) -> list[str]:
    # Число в тексте модели, которого нет во входных данных, — признак выдуманной характеристики.
    allowed = {_number(match) for match in _NUMBER_RE.findall(source)}
    found: list[str] = []
    for token in _NUMBER_RE.findall(prose):
        normalized = _number(token)
        if normalized in allowed or normalized in _SAFE_NUMBERS:
            continue
        if token not in found:
            found.append(token)
    return found


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    queries = raw.get("search_queries") or []
    if isinstance(queries, str):
        queries = queries.splitlines()
    if not isinstance(queries, list):
        queries = []
    return {
        "seo_title": _as_text(raw.get("seo_title")),
        "seo_description": _as_text(raw.get("seo_description")),
        "search_queries": [
            text for text in (_as_text(item).strip(" -•\t") for item in queries) if text
        ],
        "commercial_block": _as_text(raw.get("commercial_block")),
    }


def validate(card: dict[str, Any], source: str) -> list[tuple[str, str]]:
    title = card["seo_title"]
    description = card["seo_description"]
    commercial = card["commercial_block"]
    queries = card["search_queries"]
    issues: list[tuple[str, str]] = []

    if not title:
        issues.append(("seo_title_empty", "seo_title пустой — заполни его."))
    elif len(title) > TITLE_MAX:
        issues.append(
            (
                "seo_title_too_long",
                f"seo_title длиной {len(title)} символов, нужно не более {TITLE_MAX}.",
            )
        )

    if not description:
        issues.append(
            ("seo_description_empty", "seo_description пустой — заполни его.")
        )
    elif not DESCRIPTION_MIN <= len(description) <= DESCRIPTION_MAX:
        issues.append(
            (
                "seo_description_length",
                (
                    f"seo_description длиной {len(description)} символов, нужно от {DESCRIPTION_MIN} "
                    f"до {DESCRIPTION_MAX}. Перепиши описание нужного объёма."
                ),
            )
        )

    if not commercial:
        issues.append(
            ("commercial_block_empty", "commercial_block пустой — заполни его.")
        )

    if not QUERIES_MIN <= len(queries) <= QUERIES_MAX:
        issues.append(
            (
                "search_queries_count",
                f"Поисковых запросов {len(queries)}, нужно от {QUERIES_MIN} до {QUERIES_MAX}.",
            )
        )

    unverified = unverified_numbers(f"{title} {description} {commercial}", source)
    if unverified:
        issues.append(
            (
                "unverified_values",
                f"В тексте есть значения, которых нет во входных данных: {', '.join(unverified)}. Убери их.",
            )
        )
    return issues


def cut_at_word(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    candidate = text[:limit]
    space = candidate.rfind(" ")
    if space > limit * 0.6:
        candidate = candidate[:space]
    return candidate.rstrip(" ,.;:—–-")


def cut_at_sentence(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    candidate = text[:limit]
    position = max(candidate.rfind(". "), candidate.rfind("! "), candidate.rfind("? "))
    if position > limit * 0.6:
        return candidate[: position + 1].strip()
    space = candidate.rfind(" ")
    if space > 0:
        candidate = candidate[:space]
    return candidate.rstrip(" ,;:—–-")


def finalize(
    card: dict[str, Any], specs: list[tuple[str, str]], source: str
) -> tuple[GeneratedCard, list[WarningItem]]:
    warnings: list[WarningItem] = []

    title = card["seo_title"]
    if len(title) > TITLE_MAX:
        shortened = cut_at_word(title, TITLE_MAX)
        warnings.append(
            WarningItem(
                code="title_truncated",
                message=f"SEO-заголовок обрезан с {len(title)} до {len(shortened)} символов (лимит {TITLE_MAX}).",
            )
        )
        title = shortened

    description = card["seo_description"]
    if len(description) > DESCRIPTION_MAX:
        shortened = cut_at_sentence(description, DESCRIPTION_MAX)
        warnings.append(
            WarningItem(
                code="description_truncated",
                message=f"SEO-описание обрезано с {len(description)} до {len(shortened)} символов "
                f"(лимит {DESCRIPTION_MAX}).",
            )
        )
        description = shortened
    if len(description) < DESCRIPTION_MIN:
        warnings.append(
            WarningItem(
                code="description_too_short",
                message=f"SEO-описание короче нормы: {len(description)} символов при минимуме {DESCRIPTION_MIN}.",
            )
        )

    queries = card["search_queries"]
    if len(queries) > QUERIES_MAX:
        warnings.append(
            WarningItem(
                code="search_queries_trimmed",
                message=f"Поисковых запросов было {len(queries)}, оставлены первые {QUERIES_MAX}.",
            )
        )
        queries = queries[:QUERIES_MAX]
    if len(queries) < QUERIES_MIN:
        warnings.append(
            WarningItem(
                code="search_queries_insufficient",
                message=f"Модель предложила {len(queries)} поисковых запросов при минимуме {QUERIES_MIN}.",
            )
        )

    commercial = card["commercial_block"]
    if not commercial:
        warnings.append(
            WarningItem(
                code="commercial_block_empty",
                message="Модель не заполнила commercial-блок.",
            )
        )

    unverified = unverified_numbers(f"{title} {description} {commercial}", source)
    if unverified:
        warnings.append(
            WarningItem(
                code="unverified_values",
                message="В тексте остались значения, которых нет во входных данных: "
                f"{', '.join(unverified)}. Проверьте вручную.",
            )
        )

    generated = GeneratedCard(
        seo_title=title,
        seo_description=description,
        characteristics=characteristics(specs),
        search_queries=queries,
        commercial_block=commercial,
    )
    return generated, warnings


def _user_message(payload: dict[str, Any], missing: list[str]) -> str:
    return (
        "Данные товара (JSON, единственный источник фактов):\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"Отсутствующие характеристики (в тексте их не упоминать): {', '.join(missing) or 'нет'}.\n\n"
        "Сформируй черновик карточки. Ответ — только JSON по схеме."
    )


def _correction(issues: list[tuple[str, str]]) -> str:
    bullets = "\n".join(f"- {message}" for _, message in issues)
    return (
        "Предыдущий ответ не прошёл проверку:\n"
        f"{bullets}\n\n"
        "Перепиши ответ полностью, устранив проблемы. Фразы со значениями, которых нет во входных "
        "данных, удали целиком, а не заменяй другими числами. Если нарушен объём seo_description — "
        "разверни его до трёх абзацев по 4–5 предложений (назначение, конструкция и условия "
        "эксплуатации, задачи и аудитория) без новых фактов. Ответ — только JSON по схеме."
    )


async def _chat(messages: list[dict[str, str]]) -> dict[str, Any]:
    client = ollama.AsyncClient(host=OLLAMA_HOST, timeout=TIMEOUT)
    try:
        response = await client.chat(
            model=MODEL,
            messages=messages,
            # qwen3 — thinking-модель: без think=False рассуждения уходят в отдельное поле, а content пустой
            think=False,
            format=CARD_SCHEMA,
            options={
                "temperature": 0.2,
                "top_p": 0.9,
                "num_ctx": 8192,
                "num_predict": 1600,
            },
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504, detail=f"Модель {MODEL} не ответила за {TIMEOUT:.0f} с"
        )
    except ollama.ResponseError as exc:
        raise HTTPException(
            status_code=502, detail=f"Ollama вернул ошибку: {exc.error}"
        )
    except (httpx.HTTPError, ConnectionError):
        raise HTTPException(
            status_code=502, detail=f"Ollama недоступен по адресу {OLLAMA_HOST}"
        )
    finally:
        await client.close()

    content = response.message.content or ""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise HTTPException(
            status_code=502, detail=f"Модель вернула не-JSON: {content[:200]}"
        )


async def generate_card(product: ProductIn) -> CardResponse:
    started = time.perf_counter()
    specs, missing = prepare(product)
    description = product.description.strip()[:DESCRIPTION_LIMIT]
    payload = {
        "name": product.name,
        "manufacturer": product.manufacturer,
        "specs": dict(specs),
        "description": description,
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_message(payload, missing)},
    ]
    source = " ".join(
        [product.name, product.manufacturer, description, *product.specs.values()]
    )

    attempts = 0
    card: dict[str, Any] = {}
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        card = normalize(await _chat(messages))
        issues = validate(card, source)
        if not issues or attempts == MAX_ATTEMPTS:
            break
        messages += [
            {"role": "assistant", "content": json.dumps(card, ensure_ascii=False)},
            {"role": "user", "content": _correction(issues)},
        ]

    if not card.get("seo_title") and not card.get("seo_description"):
        raise HTTPException(status_code=502, detail="Модель вернула пустые SEO-поля")

    generated, warnings = finalize(card, specs, source)
    status = (
        "needs_review" if warnings else ("completed_with_gaps" if missing else "ok")
    )
    return CardResponse(
        status=status,
        product=ProductRef(name=product.name, manufacturer=product.manufacturer),
        card=generated,
        missing_data=missing,
        warnings=warnings,
        meta=CardMeta(
            model=MODEL,
            latency_ms=int((time.perf_counter() - started) * 1000),
            attempts=attempts,
            char_counts={
                "seo_title": len(generated.seo_title),
                "seo_description": len(generated.seo_description),
            },
        ),
    )


async def ollama_status() -> dict[str, Any]:
    client = ollama.AsyncClient(host=OLLAMA_HOST, timeout=3)
    try:
        response = await client.list()
    except (httpx.HTTPError, ollama.ResponseError, ConnectionError) as exc:
        return {
            "reachable": False,
            "model_available": False,
            "models": [],
            "error": str(exc),
        }
    finally:
        await client.close()

    models = [item.model or "" for item in response.models]
    available = MODEL in models or f"{MODEL}:latest" in models
    return {
        "reachable": True,
        "model_available": available,
        "models": models,
        "error": None,
    }
