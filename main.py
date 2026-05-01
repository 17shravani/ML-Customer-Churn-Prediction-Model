"""
ChurnGuard AI — Main Entry Point
Run: python main.py
"""
import os, sys, subprocess

def main():
    print("=" * 55)
    print("  🛡️  ChurnGuard AI — Customer Churn Prediction")
    print("=" * 55)
    print("\nChoose an action:")
    print("  1. Generate synthetic data")
    print("  2. Train model (full pipeline)")
    print("  3. Launch Streamlit Dashboard")
    print("  4. Launch FastAPI Scoring API")
    print("  5. Run all (data → train → dashboard)")
    choice = input("\n➤ Enter choice [1-5]: ").strip()

    base = os.path.dirname(__file__)

    if choice == "1":
        subprocess.run([sys.executable, os.path.join(base,"data","generate_data.py")], cwd=base)
    elif choice == "2":
        subprocess.run([sys.executable, os.path.join(base,"train.py")], cwd=base)
    elif choice == "3":
        subprocess.run(["streamlit","run","dashboard.py"], cwd=base)
    elif choice == "4":
        subprocess.run(["uvicorn","serving.app:app","--host","0.0.0.0","--port","8000","--reload"], cwd=base)
    elif choice == "5":
        print("\n[1/3] Generating data...")
        subprocess.run([sys.executable, os.path.join(base,"data","generate_data.py")], cwd=base)
        print("\n[2/3] Training model...")
        subprocess.run([sys.executable, os.path.join(base,"train.py")], cwd=base)
        print("\n[3/3] Launching dashboard...")
        subprocess.run(["streamlit","run","dashboard.py"], cwd=base)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
