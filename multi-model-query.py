#!/usr/bin/env python3
"""
Multi-Model Query Tool
Sends the same prompt to all Bedrock models and generates a markdown report.
Uses only standard library (concurrent.futures for parallelism).
"""

import argparse
import os
import urllib.request
import urllib.error
import json
import ssl
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

BEDROCK_TOKEN = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")

# All models with their regions
MODELS = {
    # Anthropic
    "Claude Opus 4.5": {
        "id": "global.anthropic.claude-opus-4-5-20251101-v1:0",
        "region": "us-east-1",
        "provider": "Anthropic"
    },
    "Claude Sonnet 4.5": {
        "id": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "region": "us-east-1",
        "provider": "Anthropic"
    },
    "Claude Haiku 4.5": {
        "id": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "region": "us-east-1",
        "provider": "Anthropic"
    },
    # OpenAI
    "GPT OSS 120B": {
        "id": "openai.gpt-oss-120b-1:0",
        "region": "us-east-1",
        "provider": "OpenAI"
    },
    # Qwen (best reasoning + best coder)
    "Qwen3 235B A22B": {
        "id": "qwen.qwen3-235b-a22b-2507-v1:0",
        "region": "us-east-2",  # ⚠️ requires us-east-2
        "provider": "Qwen"
    },
    "Qwen3 Coder 480B": {
        "id": "qwen.qwen3-coder-480b-a35b-v1:0",
        "region": "us-east-2",  # ⚠️ requires us-east-2
        "provider": "Qwen"
    },
    # Google (best only)
    "Gemma 3 27B": {
        "id": "google.gemma-3-27b-it",
        "region": "us-east-1",
        "provider": "Google"
    },
    # NVIDIA (best only)
    "Nemotron Nano 12B VL": {
        "id": "nvidia.nemotron-nano-12b-v2",
        "region": "us-east-1",
        "provider": "NVIDIA"
    },
    # Others
    "Moonshot Kimi K2": {
        "id": "moonshot.kimi-k2-thinking",
        "region": "us-east-1",
        "provider": "Moonshot"
    },
    "MiniMax M2": {
        "id": "minimax.minimax-m2",
        "region": "us-east-1",
        "provider": "MiniMax"
    },
    "DeepSeek V3.1": {
        "id": "deepseek.v3-v1:0",
        "region": "us-east-2",  # ⚠️ requires us-east-2
        "provider": "DeepSeek"
    },
}


def query_model(model_name: str, model_info: dict, prompt: str, max_tokens: int = 8192) -> dict:
    """Query a single model and return the result."""
    url = f"https://bedrock-runtime.{model_info['region']}.amazonaws.com/model/{model_info['id']}/converse"

    headers = {
        "Authorization": f"Bearer {BEDROCK_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens}
    }

    start_time = datetime.now()

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        # Create SSL context
        ctx = ssl.create_default_context()

        with urllib.request.urlopen(req, timeout=300, context=ctx) as response:
            elapsed = (datetime.now() - start_time).total_seconds()
            response_data = json.loads(response.read().decode("utf-8"))

            # Extract text from response - handle different formats
            content_list = response_data.get("output", {}).get("message", {}).get("content", [])

            text_parts = []
            reasoning_parts = []

            for content_item in content_list:
                # Check for direct text response
                if "text" in content_item:
                    text_parts.append(content_item["text"])
                # Check for reasoning content (thinking models like Kimi K2, MiniMax M2)
                elif "reasoningContent" in content_item:
                    reasoning_text = content_item.get("reasoningContent", {}).get("reasoningText", {}).get("text", "")
                    if reasoning_text:
                        reasoning_parts.append(reasoning_text)

            # Prefer direct text, fall back to reasoning
            if text_parts:
                text = "\n".join(text_parts)
            elif reasoning_parts:
                text = "[Reasoning]\n" + "\n".join(reasoning_parts)
            else:
                text = "No response text"

            # Extract token usage if available
            usage = response_data.get("usage", {})
            input_tokens = usage.get("inputTokens", "N/A")
            output_tokens = usage.get("outputTokens", "N/A")

            return {
                "model": model_name,
                "provider": model_info["provider"],
                "model_id": model_info["id"],
                "region": model_info["region"],
                "status": "success",
                "response": text,
                "elapsed_seconds": round(elapsed, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }

    except urllib.error.HTTPError as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        error_body = e.read().decode("utf-8") if e.fp else str(e)
        return {
            "model": model_name,
            "provider": model_info["provider"],
            "model_id": model_info["id"],
            "region": model_info["region"],
            "status": "error",
            "response": f"HTTP {e.code}: {error_body}",
            "elapsed_seconds": round(elapsed, 2)
        }

    except urllib.error.URLError as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        return {
            "model": model_name,
            "provider": model_info["provider"],
            "model_id": model_info["id"],
            "region": model_info["region"],
            "status": "error",
            "response": f"URL Error: {e.reason}",
            "elapsed_seconds": round(elapsed, 2)
        }

    except TimeoutError:
        return {
            "model": model_name,
            "provider": model_info["provider"],
            "model_id": model_info["id"],
            "region": model_info["region"],
            "status": "timeout",
            "response": "Request timed out after 300 seconds",
            "elapsed_seconds": 300
        }

    except Exception as e:
        return {
            "model": model_name,
            "provider": model_info["provider"],
            "model_id": model_info["id"],
            "region": model_info["region"],
            "status": "error",
            "response": str(e),
            "elapsed_seconds": 0
        }


def query_all_models(prompt: str, max_tokens: int = 8192, selected_models: Optional[list] = None) -> list:
    """Query all models concurrently using thread pool."""
    models_to_query = MODELS
    if selected_models:
        models_to_query = {k: v for k, v in MODELS.items() if k in selected_models}

    print(f"Querying {len(models_to_query)} models in parallel...")

    results = []
    with ThreadPoolExecutor(max_workers=len(models_to_query)) as executor:
        futures = {
            executor.submit(query_model, name, info, prompt, max_tokens): name
            for name, info in models_to_query.items()
        }

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                result = future.result()
                status_icon = "✅" if result["status"] == "success" else "❌"
                print(f"  {status_icon} {model_name}: {result['status']} ({result.get('elapsed_seconds', 0)}s)")
                results.append(result)
            except Exception as e:
                print(f"  ❌ {model_name}: Exception - {e}")
                results.append({
                    "model": model_name,
                    "provider": models_to_query[model_name]["provider"],
                    "model_id": models_to_query[model_name]["id"],
                    "region": models_to_query[model_name]["region"],
                    "status": "error",
                    "response": str(e),
                    "elapsed_seconds": 0
                })

    return results


def generate_markdown_report(results: list, prompt: str, output_file: str) -> str:
    """Generate a markdown report from the results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Group by provider
    by_provider: dict = {}
    for r in results:
        provider = r["provider"]
        if provider not in by_provider:
            by_provider[provider] = []
        by_provider[provider].append(r)

    # Count successes/failures
    successes = sum(1 for r in results if r["status"] == "success")
    failures = len(results) - successes

    md = f"""# Multi-Model Query Results

**Generated:** {timestamp}
**Models Queried:** {len(results)} ({successes} successful, {failures} failed)

## Prompt

```
{prompt}
```

---

## Summary Table

| Model | Provider | Status | Time (s) | Tokens (in/out) |
|-------|----------|--------|----------|-----------------|
"""

    for r in sorted(results, key=lambda x: (x["provider"], x["model"])):
        status_icon = "✅" if r["status"] == "success" else "❌"
        tokens = f"{r.get('input_tokens', 'N/A')}/{r.get('output_tokens', 'N/A')}" if r["status"] == "success" else "N/A"
        md += f"| {r['model']} | {r['provider']} | {status_icon} {r['status']} | {r.get('elapsed_seconds', 'N/A')} | {tokens} |\n"

    md += "\n---\n\n## Responses by Provider\n\n"

    # Sort providers alphabetically
    for provider in sorted(by_provider.keys()):
        md += f"### {provider}\n\n"

        for r in sorted(by_provider[provider], key=lambda x: x["model"]):
            md += f"#### {r['model']}\n\n"
            md += f"**Model ID:** `{r['model_id']}`  \n"
            md += f"**Region:** `{r['region']}`  \n"
            md += f"**Status:** {r['status']}  \n"
            md += f"**Response Time:** {r.get('elapsed_seconds', 'N/A')} seconds  \n"

            if r["status"] == "success":
                md += f"**Tokens:** {r.get('input_tokens', 'N/A')} input / {r.get('output_tokens', 'N/A')} output  \n"

            md += "\n**Response:**\n\n"
            md += f"{r['response']}\n\n"
            md += "---\n\n"

    # Write to file
    Path(output_file).write_text(md, encoding="utf-8")
    print(f"\nReport saved to: {output_file}")

    return md


def main():
    parser = argparse.ArgumentParser(
        description="Send a prompt to all non-Anthropic Bedrock models and generate a report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple prompt
  python multi-model-query.py "Explain quantum computing in 3 sentences"

  # From a file
  python multi-model-query.py --file document.txt

  # Custom output file
  python multi-model-query.py "Hello" --output results.md

  # Select specific models
  python multi-model-query.py "Hello" --models "GPT OSS 120B" "DeepSeek V3.1"

  # List available models
  python multi-model-query.py --list-models
"""
    )

    parser.add_argument("prompt", nargs="?", help="The prompt to send to all models")
    parser.add_argument("--file", "-f", help="Read prompt from a file instead")
    parser.add_argument("--output", "-o", default="model-responses.md", help="Output markdown file (default: model-responses.md)")
    parser.add_argument("--max-tokens", "-t", type=int, default=8192, help="Max tokens for response (default: 8192)")
    parser.add_argument("--models", "-m", nargs="+", help="Only query specific models (by name)")
    parser.add_argument("--list-models", "-l", action="store_true", help="List all available models")

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable Models:\n")
        for name, info in sorted(MODELS.items(), key=lambda x: (x[1]["provider"], x[0])):
            region_note = " ⚠️" if info["region"] != "us-east-1" else ""
            print(f"  [{info['provider']}] {name}{region_note}")
            print(f"    ID: {info['id']}")
            print(f"    Region: {info['region']}")
            print()
        return

    # Get prompt
    if args.file:
        prompt = Path(args.file).read_text(encoding="utf-8")
    elif args.prompt:
        prompt = args.prompt
    else:
        parser.print_help()
        print("\nError: Either provide a prompt or use --file to read from a file")
        sys.exit(1)

    print(f"Prompt length: {len(prompt)} characters")

    # Run queries
    results = query_all_models(prompt, args.max_tokens, args.models)

    # Generate report
    generate_markdown_report(results, prompt, args.output)

    # Print summary
    successes = sum(1 for r in results if r["status"] == "success")
    print(f"\nDone! {successes}/{len(results)} models responded successfully.")


if __name__ == "__main__":
    main()
