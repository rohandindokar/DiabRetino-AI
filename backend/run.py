"""Runner for the Diabetic Retinopathy FastAPI backend.

    python run.py                          # default 0.0.0.0:8000
    python run.py --port 8000 --reload
"""
import argparse
import uvicorn


def main():
    p = argparse.ArgumentParser(description="Diabetic Retinopathy FastAPI server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    a = p.parse_args()

    print("=" * 60)
    print(f"Diabetic Retinopathy AI API  ->  http://{a.host}:{a.port}")
    print(f"Swagger docs                 ->  http://{a.host}:{a.port}/docs")
    print("=" * 60)
    uvicorn.run("app.main:app", host=a.host, port=a.port, reload=a.reload)


if __name__ == "__main__":
    main()
