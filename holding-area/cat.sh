#!/bin/bash

# Output file
output_file="0400-MX-words-combined.txt"

# Clear the output file if it exists
> "$output_file"

# Prefix for all files
prefix="0400-MX-words-"

# Loop from 0 to 10 to generate the ranges (00000-00999 up to 10000-10999)
for i in {0..11}; do
    start=$((i * 1000))
    end=$(((i + 1) * 1000 - 1))
    filename="${prefix}$(printf "%05d-%05d.txt" "$start" "$end")"
    
    if [[ -f "$filename" ]]; then
        cat "$filename" >> "$output_file"
        echo "Concatenated: $filename"
    else
        echo "Warning: File not found: $filename"
    fi
done

echo "Concatenation complete. Output saved to: $output_file"
