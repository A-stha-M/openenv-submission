# Copyright (c) Meta Platforms, Inc. and affiliates.
import asyncio
import os
import textwrap
import json
from typing import List, Optional
from openai import OpenAI

# Direct import guarantees state preservation
from my_env.server.my_env_environment import MyEnvironment
from my_env.models import MyEnvAction

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"

TASKS = ["cold_chain_easy", "cold_chain_medium", "cold_chain_hard"]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Logistics Dispatcher. 
    Goal: Deliver cargo to 'Destination' before temperature exceeds 5.0°C and before fuel hits 0%.
    
    CRITICAL STRATEGY:
    - If the task is 'medium' or 'hard', ambient temperatures are lethal. You MUST increase 'cooling_power' to 0.95 or 1.0.
    - If you are far from the destination, increase 'speed_kmh' to 95.0 or 100.0 to arrive before spoiling, but watch your fuel!
    
    Output ONLY valid JSON. Do not include markdown blocks like ```json.
    Format exactly like this but change the values based on the status:
    {"target_hub": "Destination", "cooling_power": <float>, "speed_kmh": <float>}
    """
).strip()

def log_start(task, env, model):
    print(f"\n[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def run_task(client, env, task_name):
    rewards = []
    steps_taken = 0
    success = False
    
    log_start(task_name, "cold_chain_logistics", MODEL_NAME)

    try:
        # Reset environment with the specific task
        obs = env.reset(task_name=task_name)
        
        for step in range(1, 15): # Give it up to 15 steps just in case
            user_prompt = f"Task: {task_name} | Status: Temp={obs.cargo_temp_celsius:.1f}C, Fuel={obs.fuel_level_percent:.1f}%, Dist={obs.distance_to_destination_km:.1f}km"
            
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"}
            )
            
            action_raw = completion.choices[0].message.content or "{}"
            action_dict = json.loads(action_raw)
            
            # Convert raw JSON to Pydantic Action and step the environment
            action = MyEnvAction(**action_dict)
            obs = env.step(action)
            
            # The observation object directly holds the reward and done state
            reward = obs.reward
            done = obs.done
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step, action_raw, reward, done, None)
            
            if done:
                success = obs.current_location == "Destination"
                break

    except Exception as e:
        log_step(steps_taken+1, "error", 0.0, True, str(e))
    finally:
        total_score = min(max(sum(rewards), 0.0), 1.0) if success else 0.0
        log_end(success, steps_taken, total_score, rewards)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = MyEnvironment() # Create exactly ONE truck
    
    for task in TASKS:
        await run_task(client, env, task)

if __name__ == "__main__":
    asyncio.run(main())