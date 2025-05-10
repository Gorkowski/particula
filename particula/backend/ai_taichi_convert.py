"""
This script calls the Aider CLI to convert a Python file to Taichi code.
It uses the Aider model to generate the code and then reflects on the changes made.
"""


prompt="$1"

# first shot cli call
aider \
    --model o3-mini \
    --architect \
    --reasoning-effort high \
    --editor-model sonnet \
    --no-detect-urls \
    --no-auto-commit \
    --yes-always \
    --file *.py \
    --message "$prompt"

# reflection cli call
aider \
    --model o3-mini \
    --architect \
    --reasoning-effort high \
    --editor-model sonnet \
    --no-detect-urls \
    --yes-auto-commit \
    --yes-always \
    --file *.py \
    --message "Double all changes requested to make sure they've been implemented: $prompt"