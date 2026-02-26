# check_files.py

import os
import sys

print("="*50)
print("CHECKING PROJECT STRUCTURE")
print("="*50)

# Check current directory
print(f"\nCurrent directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

# Check if models directory exists
models_path = 'models'
if os.path.exists(models_path):
    print(f"\n✅ Models directory found at: {models_path}")
    print(f"Files in models directory: {os.listdir(models_path)}")
else:
    print(f"\n❌ Models directory NOT found at: {models_path}")
    
    # Check if it exists in parent directory
    parent_models = '../models'
    if os.path.exists(parent_models):
        print(f"✅ Models directory found at: {parent_models}")
        print(f"Files in parent models directory: {os.listdir(parent_models)}")
    else:
        print(f"❌ Models directory NOT found at: {parent_models}")

# Check if src directory exists
src_path = 'src'
if os.path.exists(src_path):
    print(f"\n✅ Src directory found at: {src_path}")
    print(f"Files in src directory: {os.listdir(src_path)}")
else:
    print(f"\n❌ Src directory NOT found at: {src_path}")

# Check if outputs directory exists
outputs_path = 'outputs'
if os.path.exists(outputs_path):
    print(f"\n✅ Outputs directory found at: {outputs_path}")
    print(f"Files in outputs directory: {os.listdir(outputs_path)}")
else:
    print(f"\n❌ Outputs directory NOT found at: {outputs_path}")

print("\n" + "="*50)
print("RUN THESE COMMANDS TO FIX:")
print("="*50)
print("""
1. Make sure you're in the right directory:
   cd C:\\Users\\Lenovo\\Downloads\\explainable-credit-risk

2. Activate virtual environment:
   .venv\\Scripts\\activate

3. Run training from the correct location:
   cd src
   python main.py

4. After training, check if models were created:
   cd ..
   dir models

5. Run the app:
   streamlit run app.py
""")