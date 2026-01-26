import os
import shutil
import sys
import glob

# 1. Define the Target (Where llama.dll lives)
venv_base = sys.prefix
target_dir = os.path.join(venv_base, "Lib", "site-packages", "llama_cpp", "lib")

print(f"🎯 Target Folder: {target_dir}")

# 2. Find the Source (Where nvidia-*-cu12 installed the keys)
# They usually live in venv/Lib/site-packages/nvidia/.../bin or lib
source_patterns = [
    os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cuda_runtime", "bin", "*.dll"),
    os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cublas", "bin", "*.dll"),
    os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cuda_runtime", "lib", "*.dll"),  # Sometimes here
]

copied_count = 0
needed_files = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]

print("\n🕵️  Hunting for missing DLLs...")
for pattern in source_patterns:
    for dll_file in glob.glob(pattern):
        filename = os.path.basename(dll_file)

        # We only want the critical runtime DLLs
        if filename in needed_files:
            dest_path = os.path.join(target_dir, filename)

            if not os.path.exists(dest_path):
                print(f"   -> Copying {filename}...")
                shutil.copy2(dll_file, dest_path)
                copied_count += 1
            else:
                print(f"   -> {filename} already exists. Skipping.")

# 3. Final Check
print(f"\n✅ Copied {copied_count} files.")
print("   Verifying ignition keys in target folder:")
missing = []
for f in needed_files:
    if os.path.exists(os.path.join(target_dir, f)):
        print(f"   OK: {f}")
    else:
        print(f"   ❌ MISSING: {f}")
        missing.append(f)

if not missing:
    print("\n🚀 ALL SYSTEMS GO. Run testNvidia.py now.")
else:
    print("\n⚠️ Still missing files. Did 'pip install nvidia-cuda-runtime-cu12' work?")