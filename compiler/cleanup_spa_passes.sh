#!/bin/bash
# Clean up and rename SPA passes for consistency

set -e

echo "🔧 Cleaning up SPA passes..."

# 1. Rename directory if needed
if [ -d "passes/spa" ]; then
    echo "Renaming passes/spa/ to passes/sparseflow/"
    mv passes/spa passes/sparseflow
fi

# 2. Rename files
for file in passes/sparseflow/*; do
    if [[ $file == *"spa"* ]]; then
        newfile=$(echo $file | sed 's/spa/sparseflow/g')
        echo "Renaming: $file → $newfile"
        mv "$file" "$newfile"
    fi
done

# 3. Update includes in source files
echo "Updating includes..."
find passes/sparseflow -name "*.cpp" -o -name "*.h" | while read file; do
    sed -i 's/#include "spa\//#include "sparseflow\//g' "$file"
    sed -i 's/"spa\.h"/"sparseflow.h"/g' "$file"
done

# 4. Update CMakeLists.txt if needed
if [ -f "passes/CMakeLists.txt" ]; then
    sed -i 's/spa/sparseflow/g' passes/CMakeLists.txt
fi

echo "✅ SPA pass cleanup complete!"
