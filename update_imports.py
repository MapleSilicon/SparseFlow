#!/usr/bin/env python3
import os
import re

def update_imports_in_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    original = content
    
    replacements = [
        (r'from kernel_cache import', 'from tools.kernel_cache import'),
        (r'import kernel_cache', 'import tools.kernel_cache as kernel_cache'),
        (r'kernel_selection_cache\.db', 'benchmarks/kernel_selection_cache.db'),
        (r'"kernel_selection_cache\.db"', '"benchmarks/kernel_selection_cache.db"'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

for dir_name in ['benchmarks', 'tools', 'tests', 'experiments']:
    if not os.path.exists(dir_name):
        continue
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if update_imports_in_file(filepath):
                    print(f"✓ Updated: {filepath}")

print("\n✅ Import updates complete!")
