#!/bin/bash

"""
Batch Evaluation Script
=====================

This script automates the evaluation of multiple model checkpoints
and records their BLEU scores in a CSV file for analysis.

Key features:
- Automated evaluation of multiple checkpoints
- BLEU score extraction and recording
- CSV output for easy analysis
- Error handling for missing checkpoints
- Progress tracking

Usage:
    bash batch_evaluation.sh

Requirements:
    - model_evaluation.py script
    - Checkpoint files in checkpoints/ directory
    - Test dataset files in datasets/ directory

Author: Implementation for Transformer evaluation pipeline
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

# The name of the Python script to execute for evaluation
PYTHON_SCRIPT="model_evaluation.py"

# The directory where model checkpoints are stored
CHECKPOINTS_DIR="checkpoints"

# The output file for the results (CSV format)
OUTPUT_CSV="bleu_scores.csv"

# The sequence of steps to test
# Format: START INCREMENT END
START_STEP=260000
INCREMENT=20000
END_STEP=480000

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

echo "Starting model evaluation..."
echo "Results will be saved to ${OUTPUT_CSV}"
echo "Evaluating checkpoints from step ${START_STEP} to ${END_STEP} (increment: ${INCREMENT})"
echo ""

# Create the CSV file and write the header row
# The '>' operator creates a new file or overwrites an existing one
echo "Step,BLEU_Score" > "${OUTPUT_CSV}"

# Loop through the specified range of steps
# 'seq' generates a sequence of numbers from START to END with a given INCREMENT
for step in $(seq ${START_STEP} ${INCREMENT} ${END_STEP}); do
    
    # Construct the full path to the checkpoint file for the current step
    checkpoint_file="step_${step}.pt"
    checkpoint_path="${CHECKPOINTS_DIR}/${checkpoint_file}"
    
    echo "-------------------------------------------"
    echo "Processing Step: ${step}"
    echo "Checkpoint: ${checkpoint_path}"

    # Check if the checkpoint file actually exists before trying to run it
    if [ ! -f "${checkpoint_path}" ]; then
        echo "WARNING: Checkpoint file not found at '${checkpoint_path}'. Skipping."
        # Use 'continue' to skip the rest of the loop and move to the next step
        continue
    fi
    
    # Execute the Python script and pass the checkpoint filename as an argument
    # We pipe '|' the output of the python script to other commands to process it
    # 'grep' filters the output, keeping only the line containing "Final Corpus BLEU score"
    # The result of this pipeline is stored in the 'result_line' variable
    echo "Running evaluation..."
    result_line=$(python3 "${PYTHON_SCRIPT}" "${checkpoint_file}" | grep "Final Corpus BLEU score")
    
    # Check if grep found the line. If not, the script might have failed
    if [ -z "${result_line}" ]; then
        echo "ERROR: Could not find BLEU score in the output for ${checkpoint_file}. Check for errors."
    else
        # If the line was found, parse it to extract the score
        # 'awk' is a powerful text-processing tool. We tell it to print the last field ($NF),
        # which in our case is the numerical score
        bleu_score=$(echo "${result_line}" | awk '{print $NF}')
        
        # Print the extracted score to the console for real-time feedback
        echo "Found BLEU Score: ${bleu_score}"
        
        # Append the step number and the extracted BLEU score to the CSV file
        # The '>>' operator appends to the file without overwriting it
        echo "${step},${bleu_score}" >> "${OUTPUT_CSV}"
        echo "Results saved to ${OUTPUT_CSV}"
    fi
    echo ""

done

# ============================================================================
# COMPLETION
# ============================================================================

echo "-------------------------------------------"
echo "Evaluation complete."
echo "All scores have been documented in ${OUTPUT_CSV}"
echo "You can now analyze the results using:"
echo "  - Excel or Google Sheets"
echo "  - Python pandas"
echo "  - Any CSV analysis tool"
echo ""
echo "Thank you for using the batch evaluation script!"