#!/bin/bash

# Directory containing config files
config_dir="config_files"
script_name="compare_global_features"

# Output directory for log files
log_dir="log_files"

# Create the log directory if it doesn't exist
mkdir -p "$log_dir"

# Check if the config directory exists
if [ ! -d "$config_dir" ]; then
  echo "Error: Directory $config_dir not found."
  exit 1
fi

# Loop through all config files in the directory and run the command with nohup
for config_file in "$config_dir/dift/"*.yaml
do
  # Get the filename without extension
  filename=$(basename "${config_file%.yaml}")

  # Run the command and store output and error in the log directory with nohup
  #python -u visualize_dino_features.py --config "$config_file" > "$log_dir/nohup_${filename}.out" 2> "$log_dir/nohup_${filename}.err"
  python -u $script_name.py --local_rank 1 --config "$config_file" > "$log_dir/nohup_${filename}.out" 2> "$log_dir/nohup_${filename}.err"
done
