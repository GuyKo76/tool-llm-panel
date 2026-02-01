# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

**When the user provides a question or prompt, your job is to:**
1. Save it to a `.txt` file (no modifications)
2. Run it through all Bedrock models using `multi-model-query.py`
3. Collect outputs into an organized markdown file (no modifications)

**Critical: Zero manipulation.** The prompt goes in exactly as given. Model outputs are collected exactly as returned. No summarizing, no editing, no "improvements" on either end.

## Overview

Multi-model query tool for comparing responses from various LLM providers through AWS Bedrock. Sends the same prompt to multiple models in parallel and generates a markdown report organized by model/provider.

## Typical Workflow

```bash
# 1. Save user's prompt to file (no modifications)
# 2. Run against all models
python3 multi-model-query.py --file prompt.txt --output responses.md

# 3. Results are in responses.md, organized by provider/model
```

## Commands

```bash
# Run with prompt from file (preferred)
python3 multi-model-query.py --file prompt.txt --output results.md

# Run with inline prompt
python3 multi-model-query.py "Your prompt here"

# Query specific models only
python3 multi-model-query.py "prompt" --models "Claude Opus 4.5" "DeepSeek V3.1"

# List available models
python3 multi-model-query.py --list-models

# Set max tokens (default: 8192)
python3 multi-model-query.py --file prompt.txt --max-tokens 4096
```

## Architecture

Single-file tool using only Python standard library (no pip dependencies). Uses `concurrent.futures.ThreadPoolExecutor` to query all models in parallel.

**Authentication**: Reads `AWS_BEARER_TOKEN_BEDROCK` from environment (sourced from `~/.secrets`).

**Model regions**: Most models run in `us-east-1`, but Qwen and DeepSeek models require `us-east-2`.

**Response handling**: Supports both direct text responses and reasoning/thinking model outputs (e.g., Kimi K2, MiniMax M2). Reasoning models output is prefixed with `[Reasoning]`.

**Timeout**: 300 seconds per model request.

## File Naming Convention

Files use date prefixes for chronological sorting:
```
YYYY-MM-DD-<topic>-prompt.txt     # Input prompt
YYYY-MM-DD-<topic>-responses.md   # Model outputs
```

## Available Models

Run `python3 multi-model-query.py --list-models` for current list. Models span Anthropic, OpenAI, Qwen, Google, NVIDIA, Moonshot, MiniMax, and DeepSeek.

## Output Format

Generates markdown with:
1. Summary table (model, provider, status, time, tokens)
2. Full responses grouped by provider
