"""Minimal model-call boundary.

The controller talks to models through :class:`ModelClient`. The only production implementation is
Bedrock ConverseStream. Models receive text and return text; they have no tools. Structured output is
parsed by :func:`parse_json_block`, and the caller validates the schema.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

from . import config as C

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


class ModelError(RuntimeError):
    pass


@dataclass
class ModelReply:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    model_id: str


class ModelClient(Protocol):
    def complete(self, *, role: str, model_id: str, system: str, user: str, max_tokens: int) -> ModelReply: ...


def parse_json_block(text: str) -> dict:
    """Accept a fenced ```json block or a bare top-level object; reject anything else."""
    m = JSON_BLOCK_RE.search(text)
    candidate = m.group(1) if m else text.strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ModelError("model reply contains no JSON object")
        try:
            data = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ModelError(f"model reply JSON is invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise ModelError("model reply JSON is not an object")
    return data


def inference_config(model_id: str, max_tokens: int) -> dict:
    cfg = {"maxTokens": max_tokens}
    if not (
        model_id.startswith(("us.anthropic.claude-fable-", "global.anthropic.claude-fable-"))
        or model_id.startswith(("us.openai.gpt-", "global.openai.gpt-"))
    ):
        cfg["temperature"] = 0.0
    return cfg


class BedrockClient:
    """boto3 is imported lazily so tests never need it."""

    def __init__(self, cfg: C.LoopConfig):
        try:
            import boto3  # noqa: F401
            from botocore.config import Config  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModelError("boto3 missing; install research/agentic/requirements.txt") from exc
        import boto3
        from botocore.config import Config

        profile = cfg.aws_profile or os.getenv("AWS_PROFILE") or None
        region = cfg.aws_region or os.getenv("AWS_REGION") or "us-west-2"
        session = boto3.Session(profile_name=profile, region_name=region)
        self._client = session.client(
            "bedrock-runtime",
            config=Config(connect_timeout=10, read_timeout=600, retries={"mode": "standard", "max_attempts": 2}),
        )

    def complete(self, *, role: str, model_id: str, system: str, user: str, max_tokens: int) -> ModelReply:
        from botocore.exceptions import BotoCoreError, ClientError
        from urllib3.exceptions import HTTPError

        for attempt in range(2):
            try:
                resp = self._client.converse_stream(
                    modelId=model_id,
                    system=[{"text": system}],
                    messages=[{"role": "user", "content": [{"text": user}]}],
                    inferenceConfig=inference_config(model_id, max_tokens),
                )
                # These are deliberately reset per attempt. Partial text from a truncated stream
                # must never be joined to the fresh response.
                chunks: list[str] = []
                usage: dict = {}
                stop_reason = ""
                error_events = (
                    "internalServerException",
                    "modelStreamErrorException",
                    "validationException",
                    "throttlingException",
                    "serviceUnavailableException",
                )
                for event in resp.get("stream", []):
                    for event_name in error_events:
                        if event_name in event:
                            detail = event[event_name]
                            message = (
                                detail.get("message", "unknown stream error")
                                if isinstance(detail, dict)
                                else str(detail)
                            )
                            raise ModelError(f"{event_name}: {message}")
                    delta = event.get("contentBlockDelta", {}).get("delta", {})
                    if "text" in delta:
                        chunks.append(delta["text"])
                    if "messageStop" in event:
                        stop_reason = str(event["messageStop"].get("stopReason", ""))
                    if "metadata" in event:
                        usage = event["metadata"].get("usage", {})
                return ModelReply(
                    text="".join(chunks).strip(),
                    input_tokens=int(usage.get("inputTokens", 0)),
                    output_tokens=int(usage.get("outputTokens", 0)),
                    stop_reason=stop_reason,
                    model_id=model_id,
                )
            except HTTPError as exc:
                if attempt == 0:
                    continue
                raise ModelError(f"{role} call failed after stream retry: {exc}") from exc
            except ModelError as exc:
                raise ModelError(f"{role} call failed: {exc}") from exc
            except (BotoCoreError, ClientError) as exc:
                raise ModelError(f"{role} call failed: {exc}") from exc
        raise AssertionError("unreachable")
