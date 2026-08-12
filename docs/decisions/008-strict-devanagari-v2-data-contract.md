# ADR-008: Enforce a strict Devanagari boundary for neural dataset v2

## Status

Accepted.

## Context

The v1 neural manifest described Devanagari-only records, but its eligibility
implementation accepted every Unicode letter or mark. This admitted a small
number of mixed-script and malformed tokens into training and evaluation.
They are routing/data-quality cases, not native-Hindi G2P evidence.

## Decision

Build v2 with strict Devanagari-only sentence and word filtering. A valid word
must begin with a Devanagari letter, rather than a combining mark. The builder
records the clean, split, label, and neural-manifest provenance chain, uses a
word-disjoint 90/5/5 split, and refuses to call a dataset complete unless it
has one million accepted training words.

The observed v1 blind set remains unchanged. It is not retroactively cleaned
or used to select v2 models.

## Consequences

- Mixed-script, Latin, numeric, and malformed tokens cannot enter v2 neural
  metrics or labels.
- A larger raw corpus is required because filtering and labeler rejections are
  intentional.
- The pinned legacy labeler remains the source of pronunciation/prosody labels;
  automation does not generate labels with an LLM or coding agent.
