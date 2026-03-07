import re, json
from groq import Groq
from memory import load_profile, save_profile
from prompts import build_system_prompt

client = Groq()

def extract_profile_update(text: str) -> dict | None:
    match = re.search(r'<profile_update>(.*?)</profile_update>', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
    return None

def merge_profile(existing: dict, update: dict) -> dict:
    for key, val in update.items():
        if isinstance(val, list):
            existing[key] = list(set(existing.get(key, []) + val))
        elif val:
            existing[key] = val
    return existing

def clean_response(text: str) -> str:
    text = re.sub(r'<profile_update>.*?</profile_update>', '', text, flags=re.DOTALL)
    text = text.replace("CONVERSATION_COMPLETE", "")
    return text.strip()

def run_session(user_id: str):
    profile = load_profile(user_id)
    system = build_system_prompt(profile)
    history = []

    print("\nCareer Explorer — type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            save_profile(user_id, profile)
            print("Profile saved. See you next time!")
            break

        history.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system}] + history,
                max_tokens=1024,
                timeout=10
            )
        except Exception as e:
            print(f"API Error: {e}")
            continue

        raw = response.choices[0].message.content
        update = extract_profile_update(raw)
        if update:
            profile = merge_profile(profile, update)

        reply = clean_response(raw)
        history.append({"role": "assistant", "content": raw})
        print(f"\nAgent: {reply}\n")

        if "CONVERSATION_COMPLETE" in raw:
            save_profile(user_id, profile)
            print("✅ Profile saved! Run the agent again anytime to continue exploring.\n")
            break