import asyncio
import os
import sys
import textwrap
import json
import requests
from openai import OpenAI

# ── STRICT: crash immediately if evaluator didn't inject API_KEY ──────
if not os.environ.get("API_KEY"):
    raise RuntimeError("FATAL: API_KEY environment variable is not set. Cannot proceed.")

API_KEY      = os.environ["API_KEY"]
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.environ.get("ENV_URL", "https://astha28-openenv-cold-chain.hf.space")

print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
print(f"[DEBUG] API_KEY_prefix={API_KEY[:6]}...", flush=True)
print(f"[DEBUG] ENV_URL={ENV_URL}", flush=True)

TASKS = [
    "cold_chain_easy",
    "cold_chain_medium",
    "cold_chain_hard",
    "cold_chain_vaccine_urgent",
    "cold_chain_grid_outage",
]

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI cold-chain dispatch planner.
Output ONLY valid JSON: {"target_hub": "Destination", "cooling_power": <float 0.0-1.0>, "speed_kmh": <float 40.0-120.0>}
Keep cargo below 5.0C. Do not run out of fuel. Reach Destination.
""").strip()


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={'true' if done else 'false'} error={str(error) if error else 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={'true' if success else 'false'} steps={steps} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def env_reset(task_name):
    r = requests.post(f"{ENV_URL}/reset", json={"task_name": task_name}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action_dict):
    r = requests.post(f"{ENV_URL}/step", json={"action": action_dict}, timeout=30)
    r.raise_for_status()
    return r.json()

def extract_obs(response):
    return response.get("observation", response)

def call_model(client, prompt):
    # ✅ NO try/except — crashes on any LLM failure so evaluator sees real error
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=100,
    )
    text = response.choices[0].message.content.strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in response: {text!r}")
    return json.loads(text[start:end+1])

def normalize_action(payload):
    target = payload.get("target_hub", "Destination")
    if target not in ["Destination", "Repair_Hub"]:
        target = "Destination"
    return {
        "target_hub":    target,
        "cooling_power": round(clamp(float(payload.get("cooling_power", 0.7)), 0.0, 1.0), 3),
        "speed_kmh":     round(clamp(float(payload.get("speed_kmh", 75.0)), 40.0, 120.0), 2),
    }

async def run_task(task_name):
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    rewards, steps_taken, score, success = [], 0, 0.0, False

    log_start(task_name, "cold_chain_logistics", MODEL_NAME)
    reset_resp = env_reset(task_name)
    obs = extract_obs(reset_resp)

    for step in range(1, 21):
        prompt = (
            f"Task={task_name} location={obs.get('current_location')} "
            f"temp={obs.get('cargo_temp_celsius', 2.0):.2f} "
            f"fuel={obs.get('fuel_level_percent', 100.0):.2f} "
            f"dist={obs.get('distance_to_destination_km', 0.0):.2f} "
            f"ambient={obs.get('ambient_temp_celsius', 25.0):.2f} "
            f"cooling_health={obs.get('cooling_unit_health', 1.0):.2f}"
        )

        # ✅ NO try/except — let real errors surface to evaluator logs
        action_dict = normalize_action(call_model(client, prompt))

        step_resp   = env_step(action_dict)
        obs         = extract_obs(step_resp)
        reward      = float(step_resp.get("reward", obs.get("reward", 0.0)))
        done        = bool(step_resp.get("done", obs.get("done", False)))
        score       = clamp(float(obs.get("task_score", 0.0)), 0.0, 1.0)
        rewards.append(reward)
        steps_taken = step

        log_step(step, json.dumps(action_dict), reward, done, None)
        if done:
            success = obs.get("current_location") == "Destination"
            break

    log_end(success, steps_taken, score, rewards)
    return score

async def main():
    scores = {}
    for task in TASKS:
        scores[task] = await run_task(task)
    print("\nFINAL SCORES", flush=True)
    for t, s in scores.items():
        print(f"{t}: {s:.2f}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())