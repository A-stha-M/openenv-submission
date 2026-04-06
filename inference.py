# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import asyncio
import os
import textwrap
import json
from typing import List, Optional
from openai import OpenAI

# Import our custom environment and actions
from my_env.models import MyEnvAction
from my_env.server.my_env_environment import MyEnvironment

# MANDATORY CONFIGURATION
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# Environment Metadata
TASK_NAME = os.getenv("MY_ENV_TASK", "cold_chain_easy")
BENCHMARK = "cold_chain_logistics"
MAX_STEPS = 10
TEMPERATURE = 0.3
SUCCESS_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Logistics Dispatcher managing a refrigerated delivery truck.
    Goal: Deliver cargo to 'Destination' before temperature exceeds 5.0°C.
    
    Rules:
    1. Higher 'speed_kmh' reduces travel time but burns fuel.
    2. Higher 'cooling_power' lowers cargo temp but burns fuel.
    3. If cooling health is low, you must balance speed vs. cooling.
    
    Output Format:
    Reply with ONLY a valid JSON object:
    {"target_hub": "Destination", "cooling_power": 0.8, "speed_kmh": 85.0}
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def main() -> None:
    # Initialize OpenAI Client as per mandatory instruction
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Initialize the local environment
    env = MyEnvironment()
    
    rewards: List[float] = []
    steps_taken = 0
    total_score = 0.0
    success = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Step 0: Reset Environment
        obs = env.reset()
        
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break
                
            # Build prompt for the LLM
            user_prompt = f"Status: Temp={obs.cargo_temp_celsius}C, Fuel={obs.fuel_level_percent}%, Distance={obs.distance_to_destination_km}km"
            
            # Request action from Model
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"}
            )
            
            action_raw = completion.choices[0].message.content or "{}"
            action_json = json.loads(action_raw)
            
            # Execute environment step
            action = MyEnvAction(**action_json)
            obs = env.step(action)
            
            reward = obs.reward or 0.0
            rewards.append(reward)
            steps_taken = step
            
            # MANDATORY LOGGING FORMAT
            log_step(step=step, action=action_raw, reward=reward, done=obs.done, error=None)
            
            if obs.done:
                break

        # Calculate final score (normalized 0.0 to 1.0)
        total_score = sum(rewards)
        total_score = min(max(total_score, 0.0), 1.0)
        success = (obs.current_location == "Destination") and (total_score >= SUCCESS_THRESHOLD)

    except Exception as e:
        # error=msg in log_step if an exception occurs during the loop
        log_step(step=steps_taken+1, action="error", reward=0.0, done=True, error=str(e))
    finally:
        log_end(success=success, steps=steps_taken, score=total_score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())