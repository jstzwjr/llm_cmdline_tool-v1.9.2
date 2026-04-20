#!/usr/bin/env bash

set -e

TEMP_FILELIST_PATH="clang_format_list_${RANDOM}.txt"

# Create empty temporary file
> "$TEMP_FILELIST_PATH"

find -type f -not -path './.git/*' -not -path '*/third_party/*' -not -path '*/backend/api/*' \( -name '*.h' -o -name '*.cpp' \) > "$TEMP_FILELIST_PATH"

# Begin formatting the source files
clang-format -i --verbose --files="$TEMP_FILELIST_PATH"

# Delete the temporary file
rm "$TEMP_FILELIST_PATH"