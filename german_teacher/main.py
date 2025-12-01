import asyncio
import os
from pathlib import Path

from google.adk.runners import Runner
from google.adk.apps.app import App
from google.adk.sessions import DatabaseSessionService, VertexAiSessionService
from google.adk.memory import InMemoryMemoryService
from google.genai.types import Content, Part
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv(Path(__file__).parent / ".env")

from agent import root_agent

SESSIONS_DIR = Path(os.path.expanduser("~")) / ".adk" / "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)
DB_FILE = SESSIONS_DIR / "adk_cli_sessions.db"
SESSION_URL = f"sqlite+aiosqlite:///{DB_FILE}"
MY_USER_ID = "local_cli_user_001"
MY_SESSION_ID = f"{MY_USER_ID}_cli_session"


async def main():
    print("🤖 Initializing German Teacher Agent CLI...")
    
    teacher_app = App(
        name="GermanTeacherApp",
        root_agent=root_agent,
    )

    # Check if we are running in a cloud environment with Agent Engine
    agent_engine_id = os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_ID")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if agent_engine_id and project_id:
        print(f"☁️  Using Vertex AI Agent Engine Session Service (ID: {agent_engine_id})")
        session_service = VertexAiSessionService(
            project_id=project_id,
            location=location,
            engine_id=agent_engine_id
        )
        session = await session_service.get_session(
            app_name=teacher_app.name, user_id=MY_USER_ID, session_id=MY_SESSION_ID
        )
        if session is None:
             session = await session_service.create_session(
                app_name=teacher_app.name, user_id=MY_USER_ID, session_id=MY_SESSION_ID
            )

    else:
        print(f"🗄️  Using Local Database Session Service at: {DB_FILE}")
        session_service = DatabaseSessionService(db_url=SESSION_URL)
        session = await session_service.get_session(
            app_name=teacher_app.name, user_id=MY_USER_ID, session_id=MY_SESSION_ID
        )
        if session is None:
            session = await session_service.create_session(
                app_name=teacher_app.name, user_id=MY_USER_ID, session_id=MY_SESSION_ID
            )
    
    print("--------------------------------------------------")
    print(f"✅ Session '{session.id}' is ready for user '{session.user_id}'.")

    memory_service = InMemoryMemoryService()

    runner = Runner(
        app=teacher_app,
        session_service=session_service,
        memory_service=memory_service,
    )

    while True:
        try:
            query = input("You: ")
            if query.lower() in ["quit", "exit"]:
                print("🤖 Goodbye!")
                break
            
            print("Agent: ", end="", flush=True)

            async for event in runner.run_async(
                user_id=session.user_id,
                session_id=session.id,
                new_message=Content(parts=[Part(text=query)], role="user")
            ):
                if not event.content or not event.content.parts:
                    continue

                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        print(part.text, end="", flush=True)

                    elif event.author == "user" and hasattr(part, "function_response"):
                        # Re-print the "Agent:" prompt to show it's processing the tool result
                        print("Agent: ", end="", flush=True)
            
            print("\n")

        except (KeyboardInterrupt, EOFError):
            print("\n🤖 Goodbye!")
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutting down.")
