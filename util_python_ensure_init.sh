#!/bin/bash

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No directory provided"
    echo "Usage: $0 <directory>"
    exit 1
fi

DIRECTORY="$1"

# Check if directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory '$DIRECTORY' does not exist"
    exit 1
fi

echo "Ensuring __init__.py files exist in: $DIRECTORY"
echo ""

# Counter for created files
CREATED_COUNT=0

# Function to ensure __init__.py exists in each subdirectory
ensure_init_file() {
    local current_dir="$1"
    for dir in "$current_dir"/*/; do
        if [ -d "$dir" ]; then
            # Check if __init__.py exists
            if [ ! -f "$dir/__init__.py" ]; then
                touch "$dir/__init__.py"
                CREATED_COUNT=$((CREATED_COUNT + 1))
                echo "  Created: $dir/__init__.py"
            fi
            # Recursively check subdirectories
            ensure_init_file "$dir"
        fi
    done
}

# Start the process
ensure_init_file "$DIRECTORY"

echo ""
if [ $CREATED_COUNT -eq 0 ]; then
    echo "✓ All __init__.py files already exist"
else
    echo "✓ Created $CREATED_COUNT __init__.py file(s)"
fi

