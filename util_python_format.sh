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

echo "Formatting Python code in: $DIRECTORY"
echo ""

# Remove unused imports
echo "Step 1/3: Removing unused imports with autoflake..."
autoflake -i -r --remove-all-unused-imports "$DIRECTORY"
if [ $? -eq 0 ]; then
    echo "✓ Unused imports removed"
else
    echo "✗ Error removing unused imports"
    exit 1
fi
echo ""

# Sort imports
echo "Step 2/3: Sorting imports with isort..."
isort "$DIRECTORY"
if [ $? -eq 0 ]; then
    echo "✓ Imports sorted"
else
    echo "✗ Error sorting imports"
    exit 1
fi
echo ""

# Format code
echo "Step 3/3: Formatting code with black..."
black "$DIRECTORY"
if [ $? -eq 0 ]; then
    echo "✓ Code formatted"
else
    echo "✗ Error formatting code"
    exit 1
fi
echo ""

echo "✓ All formatting steps completed successfully!"

