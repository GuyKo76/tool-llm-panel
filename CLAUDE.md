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

**Model regions**: Most models run in `us-east-1`, but Qwen and DeepSeek models require `us-east-2`.

**Response handling**: Supports both direct text responses and reasoning/thinking model outputs (e.g., Kimi K2, MiniMax M2).

## Available Models

- **Anthropic**: Claude Opus 4.5, Sonnet 4.5, Haiku 4.5
- **OpenAI**: GPT OSS 120B
- **Qwen**: Qwen3 235B, Qwen3 Coder 480B (us-east-2)
- **Google**: Gemma 3 27B
- **NVIDIA**: Nemotron Nano 12B
- **Others**: Moonshot Kimi K2, MiniMax M2, DeepSeek V3.1 (us-east-2)

## Output Format

Generates markdown with:
1. Summary table (model, provider, status, time, tokens)
2. Full responses grouped by provider
