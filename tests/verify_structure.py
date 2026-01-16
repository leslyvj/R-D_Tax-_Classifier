import os
import sys
import importlib

def check_path(path, description):
    if os.path.exists(path):
        print(f"[OK] {description} exists: {path}")
        return True
    else:
        print(f"[FAIL] {description} missing: {path}")
        return False

def verify_structure():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Verifying structure in: {base_dir}")

    required_paths = [
        "src/app/main.py",
        "src/web/streamlit_app.py",
        "data/raw",
        "data/processed/traces",
        "config",
        "docs/phase_4",
        "tests"
    ]

    all_ok = True
    for p in required_paths:
        if not check_path(os.path.join(base_dir, p), p):
            all_ok = False
            
    # Verify imports
    sys.path.append(base_dir)
    try:
        import src.app.main
        print("[OK] Import src.app.main successful")
    except ImportError as e:
        print(f"[FAIL] Import src.app.main failed: {e}")
        all_ok = False

    if all_ok:
        print("\nVerification PASSED")
    else:
        print("\nVerification FAILED")

if __name__ == "__main__":
    verify_structure()
