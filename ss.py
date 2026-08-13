import subprocess
sys_executable = subprocess.sys.executable

def setup_environment():
    print("🚀 Starting automated installation inside venv...")
    
    packages = [
        "pandas", 
        "requests", 
        "feedparser", 
        "textblob", 
        "spacy", 
        "scikit-learn", 
        "matplotlib", 
        "joblib"
    ]
    
    for pkg in packages:
        print(f"📦 Installing {pkg}...")
        subprocess.check_call([sys_executable, "-m", "pip", "install", pkg])
            
    print("📥 Downloading spaCy en_core_web_md model...")
    try:
        subprocess.check_call([sys_executable, "-m", "spacy", "download", "en_core_web_md"])
        print("✅ SpaCy model installed successfully!")
    except Exception as e:
        print(f"⚠️ Error downloading spaCy model: {e}")

    print("✨ All dependencies and models are successfully installed in your venv!")

if __name__ == "__main__":
    setup_environment()