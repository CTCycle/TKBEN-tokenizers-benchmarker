from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from server.repositories.schemas.models import MetricType

###############################################################################
def seed_metric_types(
    engine: Any,
    metric_catalog: Sequence[Mapping[str, Any]],
    *,
    commit: bool = True,
) -> None:
    entries: list[dict[str, str]] = []
    for category in metric_catalog:
        category_key = str(category.get("category_key", "uncategorized"))
        metrics = category.get("metrics")
        if not isinstance(metrics, list):
            continue
        for metric in metrics:
            if not isinstance(metric, Mapping):
                continue
            key = metric.get("key")
            label = metric.get("label")
            if not key or not label:
                continue
            entries.append(
                {
                    "key": str(key),
                    "category": category_key,
                    "label": str(label),
                    "description": str(metric.get("description") or ""),
                    "scope": str(metric.get("scope") or "aggregate"),
                    "value_kind": str(metric.get("value_kind") or "number"),
                }
            )

    if not entries:
        return

    metric_keys = [entry["key"] for entry in entries]
    with Session(bind=engine) as session:
        try:
            existing_types = {
                metric_type.key: metric_type
                for metric_type in session.execute(
                    select(MetricType).where(MetricType.key.in_(metric_keys))
                ).scalars()
            }
            for entry in entries:
                metric_type = existing_types.get(entry["key"])
                if metric_type is None:
                    session.add(MetricType(**entry))
                    continue
                metric_type.category = entry["category"]
                metric_type.label = entry["label"]
                metric_type.description = entry["description"]
                metric_type.scope = entry["scope"]
                metric_type.value_kind = entry["value_kind"]
            if commit:
                session.commit()
            else:
                session.flush()
        except Exception:
            session.rollback()
            raise
