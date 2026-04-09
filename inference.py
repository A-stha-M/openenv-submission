# Copyright (c) Meta Platforms, Inc. and affiliates.
import asyncio
import os
import sys
import textwrap
import json
from typing import Dict, List

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from my_env.server.my_env_environment import MyEnvironment
from my_env.models import MyEnvAction


TASKS = [
    "cold_chain_easy",
    "cold_chain_medium",
    "cold_chain_hard",
    "cold_chain_vaccine_urgent",
    "cold_chain_grid_outage",
]


SYSTEM_PROMPT = textwrap.dedent(
    """
You are an AI cold-chain dispatch planner.
Objective: maximize final task_score while safely reaching Destination.

Environment mechanics:
- Ambient heat and traffic change each step.
- Excessive cooling can damage compressor health and increase fuel burn.
- Thermal excursion hours are tracked against a scenario-specific compliance budget.
- Repair_Hub can restore cooling health/fuel, but route switching and repeated hub visits are penalized.
- Task score blends completion, cargo quality, fuel, schedule, discipline, and compliance.

Strategy hints:
- Keep cargo in ~1.4C to 4.2C zone and protect compliance budget.
- Prefer smooth policy (avoid frequent target switching).
- Use Repair_Hub only when risk is rising (hot cargo + low cooling health + long distance left).

Output ONLY valid JSON:
{"target_hub": "Destination", "cooling_power": <float>, "speed_kmh": <float>}
"""
).strip()


# ---------------- Logging ----------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error):
    error_str = str(error) if error else "null"
    done_str = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------- Helpers ----------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_action(payload, obs):

    target = payload.get("target_hub", "Destination")
    if target not in ["Destination", "Repair_Hub"]:
        target = "Destination"

    cooling = clamp(float(payload.get("cooling_power", 0.7)), 0.0, 1.0)
    speed = clamp(float(payload.get("speed_kmh", 75)), 40.0, 120.0)

    return {
        "target_hub": target,
        "cooling_power": round(cooling, 3),
        "speed_kmh": round(speed, 2),
    }


def call_model(client, model, user_prompt):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    text = response.choices[0].message.content.strip()

    if "```" in text:
        text = text.split("```")[1]
        if "```" in text:
            text = text.split("```")[0]

    start = text.find("{")
    end = text.rfind("}")

    text = text[start : end + 1]

    return json.loads(text)


# ---------------- Task Runner ----------------

async def run_task(task_name: str):

    # IMPORTANT: client INSIDE task
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
    )

    model = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    env = MyEnvironment()

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task_name, "cold_chain_logistics", model)

    obs = env.reset(task_name=task_name)

    for step in range(1, 21):

        user_prompt = (
            f"Task={task_name}; "
            f"location={obs.current_location}; "
            f"temp={obs.cargo_temp_celsius:.2f}; "
            f"fuel={obs.fuel_level_percent:.2f}; "
            f"ambient={obs.ambient_temp_celsius:.2f}; "
            f"dist_dest={obs.distance_to_destination_km:.2f}; "
            f"dist_hub={obs.distance_to_emergency_hub_km:.2f}; "
            f"cooling_health={obs.cooling_unit_health:.3f}; "
            f"deadline={obs.delivery_deadline_hours}; "
            f"elapsed={obs.hours_elapsed}; "
            f"urgency={obs.urgency_index:.3f}; "
            f"quality={obs.cargo_quality_index:.3f}; "
            f"compliance={obs.compliance_index:.3f}; "
            f"excursions={obs.excursion_hours}/{obs.excursion_budget_hours}; "
            f"switches={obs.route_switch_count}; "
            f"score={obs.task_score:.3f}"
        )

        # ALWAYS CALL LLM (MANDATORY)
        try:
            parsed = call_model(client, model, user_prompt)
            action_dict = normalize_action(parsed, obs)
            error = None

        except Exception as e:
            action_dict = {
                "target_hub": "Destination",
                "cooling_power": 0.75,
                "speed_kmh": 75,
            }
            error = str(e)

        action = MyEnvAction(**action_dict)
        obs = env.step(action)

        reward = obs.reward
        done = obs.done
        score = clamp(float(obs.task_score), 0.0, 1.0)

        rewards.append(reward)
        steps_taken = step

        log_step(
            step,
            json.dumps(action_dict),
            reward,
            done,
            error,
        )

        if done:
            success = obs.current_location == "Destination"
            break

    log_end(success, steps_taken, score, rewards)

    return score


# ---------------- Main ----------------

async def main():

    scores = {}

    for task in TASKS:
        score = await run_task(task)
        scores[task] = score

    print("\nFINAL SCORES", flush=True)
    for t, s in scores.items():
        print(f"{t}: {s:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
