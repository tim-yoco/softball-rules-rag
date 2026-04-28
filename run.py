#!/usr/bin/env python3
import os
import subprocess
import sys
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("Export it or add it to a .env file:")
        print("  export ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    if not os.path.exists(CHROMA_DIR):
        print("Vector store not found. Running ingestion...")
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "ingest.py")], check=True)

    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("\n" + "=" * 50)
    print("  Softball Rules Assistant")
    print("=" * 50)
    print(f"  Local:   http://localhost:8000")
    print(f"  Phone:   http://{local_ip}:8000")
    print("=" * 50 + "\n")

    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
