"""
Runner script for Diabetic Retinopathy Backend API.
Usage:
    python run.py
    python run.py --host 0.0.0.0 --port 8000
"""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start Diabetic Retinopathy FastAPI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable live auto-reload for dev")
    args = parser.parse_args()

    print(f"============================================================")
    print(f"Starting Diabetic Retinopathy AI API Server on http://{args.host}:{args.port}")
    print(f"Swagger Documentation: http://{args.host}:{args.port}/docs")
    print(f"============================================================")

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
