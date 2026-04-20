from huggingface_hub import login

if __name__ == "__main__":
    print("🔐 Logging in to Hugging Face Hub...")
    print("Visit https://huggingface.co/settings/tokens to get your token")
    print()
    login()
    print("✅ Logged in successfully!")
