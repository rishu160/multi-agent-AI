"""
eval/eval.py — Offline evaluation harness.

Run: python eval/eval.py
Requires the FastAPI server to be running on localhost:8000.
"""
import time
import json
import statistics
import httpx

API_URL = "http://localhost:8000/chat"

# ── Test suite ────────────────────────────────────────────────────────────────
# Each entry: (query, expected_route, success_keywords)
TEST_CASES = [
    # Research queries
    ("What is the latest version of Python?", "research", ["3."]),
    ("Who won the 2024 US presidential election?", "research", ["trump", "harris", "president"]),
    ("What are the main features of GPT-4o?", "research", ["openai", "model", "multimodal"]),
    ("Current price of Bitcoin", "research", ["$", "bitcoin", "btc"]),
    ("What happened at Google I/O 2024?", "research", ["google", "gemini", "ai"]),
    ("Latest news about SpaceX Starship", "research", ["spacex", "starship", "launch"]),

    # Code queries
    ("Calculate the first 10 Fibonacci numbers", "code", ["0", "1", "1", "2", "3"]),
    ("Write Python code to sort a list of dictionaries by a key", "code", ["sorted", "key", "lambda"]),
    ("What is 17 raised to the power of 13?", "code", ["98526125335693"]),
    ("Generate a Caesar cipher encoder in Python", "code", ["def", "chr", "ord"]),
    ("Calculate compound interest: $1000 at 5% for 10 years", "code", ["1628", "1629"]),
    ("Find all prime numbers up to 50", "code", ["2", "3", "5", "7", "11", "13"]),
    ("Write a binary search function", "code", ["def", "mid", "return"]),

    # Writer queries
    ("Explain quantum entanglement in simple terms", "writer", ["quantum", "particle", "entangl"]),
    ("Write a short poem about artificial intelligence", "writer", ["ai", "mind", "machine", "think", "dream"]),
    ("Summarise the plot of 1984 by George Orwell", "writer", ["winston", "orwell", "party", "big brother"]),
    ("What are the pros and cons of remote work?", "writer", ["flexib", "productiv", "isolat", "commut"]),
    ("Explain the difference between TCP and UDP", "writer", ["tcp", "udp", "reliable", "packet"]),
    ("Write a professional email declining a job offer", "writer", ["thank", "opportunity", "decline", "offer"]),
    ("What is the SOLID principle in software engineering?", "writer", ["single", "open", "liskov", "interface", "depend"]),
]

# Approximate Claude Sonnet pricing (per 1M tokens, input/output)
INPUT_TOKEN_COST_PER_M = 3.00
OUTPUT_TOKEN_COST_PER_M = 15.00
AVG_INPUT_TOKENS = 500   # rough estimate per call
AVG_OUTPUT_TOKENS = 400


def run_query(query: str) -> tuple[dict, float]:
    start = time.perf_counter()
    resp = httpx.post(API_URL, json={"query": query}, timeout=120).json()
    latency = time.perf_counter() - start
    return resp, latency


def check_success(answer: str, keywords: list[str]) -> bool:
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)


def main():
    results = []
    print(f"\n{'─'*60}")
    print(f"{'Query':<45} {'Expected':>8} {'Got':>8} {'✓':>3} {'Lat':>6}")
    print(f"{'─'*60}")

    for query, expected_route, keywords in TEST_CASES:
        try:
            resp, latency = run_query(query)
            got_route = resp.get("route", "?")
            answer = resp.get("answer", "")
            retries = resp.get("retry_count", 0)
            routed_correctly = got_route == expected_route
            task_success = check_success(answer, keywords)

            results.append({
                "query": query,
                "expected": expected_route,
                "got": got_route,
                "routed_correctly": routed_correctly,
                "task_success": task_success,
                "latency": latency,
                "retries": retries,
            })

            status = "✅" if task_success else "❌"
            route_mark = "✓" if routed_correctly else "✗"
            short_q = query[:44]
            print(f"{short_q:<45} {expected_route:>8} {got_route:>8} {route_mark:>3} {latency:>5.1f}s {status}")

        except Exception as exc:
            print(f"ERROR on '{query[:40]}': {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results)
    routing_pct = sum(r["routed_correctly"] for r in results) / n * 100
    success_pct = sum(r["task_success"] for r in results) / n * 100
    latencies = [r["latency"] for r in results]
    avg_lat = statistics.mean(latencies)
    p90_lat = sorted(latencies)[int(0.9 * n)]
    avg_retries = statistics.mean(r["retries"] for r in results)

    # Cost estimate: assume 3 LLM calls per query (supervisor + specialist + critic)
    calls_per_query = 3 + avg_retries
    cost_per_query = (
        (AVG_INPUT_TOKENS * calls_per_query / 1_000_000 * INPUT_TOKEN_COST_PER_M)
        + (AVG_OUTPUT_TOKENS * calls_per_query / 1_000_000 * OUTPUT_TOKEN_COST_PER_M)
    )

    print(f"\n{'─'*60}")
    print(f"Total queries       : {n}")
    print(f"Correct routing     : {routing_pct:.1f}%")
    print(f"Task success        : {success_pct:.1f}%")
    print(f"Avg latency         : {avg_lat:.2f}s")
    print(f"p90 latency         : {p90_lat:.2f}s")
    print(f"Avg retries         : {avg_retries:.2f}")
    print(f"Est. cost / query   : ${cost_per_query:.5f}")
    print(f"Est. cost / 1k q    : ${cost_per_query * 1000:.3f}")
    print(f"{'─'*60}\n")

    # Save raw results
    with open("eval/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Full results saved to eval/results.json")


if __name__ == "__main__":
    main()
