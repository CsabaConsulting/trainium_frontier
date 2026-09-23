# Bedrock transport incident — 2026-09-23

## Scope

This was a local autonomous-controller transport incident. It did not alter `train.py`, start a
training run, or change either Trn2 instance.

## Evidence

- Cycle 137 ran from `2026-09-22T22:45:14Z` to `23:02:48Z` and the non-streaming Fable 5.1
  `Converse` request ended with `Connection was closed before we received a valid response`.
- A minimal authenticated Fable 5.1 invocation subsequently succeeded, showing that the bearer
  token, model entitlement, region, and short request path were healthy.
- Cycle 152 retried from `2026-09-22T23:32:48Z` to `23:48:11Z` and ended with the same connection
  close. Socket inspection during both cycles showed an established TLS connection and a retry,
  while the large response remained silent.

The working diagnosis is that a long non-streaming response left the HTTP connection idle long
enough for an intermediary to close it. This is an inference from the timing and socket evidence;
it is not presented as an AWS service root-cause determination.

## Repair iterations

`research/agentic/autoloop/models.py` now uses Bedrock `ConverseStream` and:

- concatenates text deltas;
- captures terminal usage and stop-reason metadata;
- fails closed on every modeled stream error event; and
- preserves the existing structured-response validation boundary.

The first full streamed call (cycle 154) showed that Fable emitted no event before the connection's
approximately five-minute idle cutoff. Its `urllib3.ProtocolError` also escaped the original
botocore-only exception boundary and terminated the local service. The follow-up repair therefore:

- converts premature HTTP stream termination into the same recoverable `ModelError` path;
- bounds the model-facing ledger view to 60 recent compact rows while retaining the full ledger on
  disk;
- bounds verbose rejection fields while retaining their proposal ids and verifier findings; and
- reduces the researcher output cap from 64K to 32K tokens. A prior 16K cap was insufficient, so
  32K retains explicit reasoning headroom while reducing time-to-first-output pressure.

Cycle 173 later proved that a healthy event stream can still terminate prematurely after emitting
partial data. The controller caught the resulting `ProtocolError` and backed off safely, but that
discarded a full research attempt. `BedrockClient` now retries a premature stream exactly once as a
fresh request. Partial chunks are reset before the retry, and a second premature termination is
reported through the existing recoverable `ModelError` path. Modeled Bedrock error events are not
retried by this wrapper.

## Verification

- `python -m unittest discover -s research/agentic/tests -v`: the complete offline suite passes,
  including streamed text, modeled stream error, premature stream termination, and bounded evidence
  tests.
- A live request through the patched `BedrockClient` returned exactly `BEDROCK_STREAM_OK` with
  `input_tokens=37`, `output_tokens=16`, and `stop_reason=end_turn`.

The full autonomous researcher call remains the production verification gate. A proposal is not
accepted, and no experiment is launched, unless all existing schema, verifier, preflight, and
run-protocol gates pass after this transport repair.
