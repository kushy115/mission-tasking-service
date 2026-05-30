"""OpenTelemetry tracing setup.

Instrument FastAPI so every compile request is a span tree. Spans propagate
into LangGraph node code, tool calls, and SQLAlchemy via the auto-instrumenters.
Exports via OTLP to the collector listed in OTEL_EXPORTER_OTLP_ENDPOINT.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger(__name__)


def configure_tracing(app: FastAPI) -> None:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    service_name = os.getenv("OTEL_SERVICE_NAME", "mission-tasking-service")
    if not endpoint:
        log.info("OTEL endpoint not set; tracing disabled.")
        return
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument()
    except Exception as e:  # noqa: BLE001
        log.warning("sqlalchemy instrumentation unavailable: %s", e)
    log.info("OTel tracing configured -> %s", endpoint)
