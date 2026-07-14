import pandas as pd
import requests
import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../backend/.env"))

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BACKEND_URL = "http://127.0.0.1:8000"
JUDGE_MODEL = "llama-3.1-8b-instant"

folder = os.path.dirname(__file__)
CSV_PATH = os.path.join(folder, "test_set.csv")
RESULTS_PATH = os.path.join(folder, "eval_results.csv")

JUDGE_PROMPT = """You are an examiner that compares generated answers and the actual correct answers.
Rules:
- Only use information from the generated answer and actual answer provided
- On the basis of comparison, provide a score between 0-5, where 0 is lowest, 3 is average and 5 is highest
- Be concise and direct
- Return only a JSON object with no extra text, no markdown, no backticks:
{"score": <number>, "reason": "<string>"}"""


def judge_answer(generated_answer: str, expected_answer: str) -> dict:
    """Uses Groq LLM to score the generated answer against the expected answer."""
    user_message = f"Expected answer: {expected_answer}\nGenerated answer: {generated_answer}"

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.0
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # If LLM added extra text despite instructions, try to extract JSON
        start = raw.find("{")
        end = raw.rfind("}") + 1
        result = json.loads(raw[start:end])

    return result


def run_evals():
    df = pd.read_csv(CSV_PATH)

    results = []

    for index, row in df.iterrows():
        print(f"\nEvaluating question {index + 1}/{len(df)}: {row['question']}")

        # Step 1: Get generated answer from RAG pipeline
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={
                "filename": row["filename"],
                "question": row["question"]
            }
        )

        if response.status_code != 200:
            print(f"Query failed: {response.text}")
            continue

        generated_answer = response.json()["answer"]
        print(f"Generated: {generated_answer[:100]}...")

        # Step 2: Judge the answer
        judgment = judge_answer(generated_answer, row["expected_answer"])
        score = judgment.get("score", 0)
        reason = judgment.get("reason", "")

        print(f"Score: {score}/5 — {reason}")

        results.append({
            "question": row["question"],
            "expected_answer": row["expected_answer"],
            "generated_answer": generated_answer,
            "score": score,
            "reason": reason,
            "filename": row["filename"]
        })

    # Step 3: Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    # Step 4: Print summary report
    avg_score = results_df["score"].mean()
    print(f"\n{'='*50}")
    print(f"EVAL REPORT")
    print(f"{'='*50}")
    print(f"Total questions: {len(results_df)}")
    print(f"Average score: {avg_score:.2f}/5")
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_evals()