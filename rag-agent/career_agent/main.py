from memory import init_db
from agent import run_session

if __name__ == "__main__":
    init_db()
    user_id = input("Enter your name: ").strip()
    run_session(user_id)