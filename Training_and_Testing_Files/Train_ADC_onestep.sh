#!/bin/bash

# =================================================================
# USAGE: ./run_estimator.sh <dataset> <dim> <time> [nan_to] [layers]
# =================================================================

# 1. Read User Inputs with Defaults
dataset_name=$1
dimension_init=$2
Time_min=$3
added_layers=$4
nan_to=${5:-"-1000000000"}


# 2. Validation
if [ -z "$Time_min" ]; then
    echo "Error: You must provide at least 3 arguments: dataset_name, dimension_init, Time_min"
    exit 1
fi

# 3. Create output AND log directories
mkdir -p pipeline_artifacts logs

echo "--- Submitting Cardinality Estimator Pipeline ---"
echo "Config: Dataset=$dataset_name, Dim=$dimension_init, Time=$Time_min, NaN=$nan_to, Layers=$added_layers"

# =================================================================
# STEP 1: Program A (Histograms)
# =================================================================
JOB_A_ID=$(bsub \
    -J "CardEst_Histograms" \
    -o "logs/%J.out" \
    -e "logs/%J.err" \
    -n 36 -R "span[hosts=1]" \
    -W 60 \
    "python Train_ADC_All_Histograms.py $dataset_name $dimension_init $nan_to" \
    | grep -oE '[0-9]+')

echo "Step 1: Submitted Job A (ID: $JOB_A_ID)"

# =================================================================
# STEP 2: The Parallel Block (B and 4x C)
# =================================================================

PARALLEL_JOBS=""

# --- Program B (GMM) ---
ID_B=$(bsub \
    -J "CardEst_GMM" \
    -w "done($JOB_A_ID)" \
    -o "logs/%J.out" \
    -e "logs/%J.err" \
    -n 36 -R "span[hosts=1]" \
    -W 600 \
    "python Train_ADC_GMM.py $dataset_name $dimension_init $nan_to" \
    | grep -oE '[0-9]+')
PARALLEL_JOBS+="done($ID_B)"

# --- Program C (Network Training - 4 Copies) ---
# Note: \$LSB_JOBINDEX is escaped so it is evaluated on the compute node
ID_C=$(bsub \
    -J "CardEst_Trainset[1-4]" \
    -w "done($JOB_A_ID)" \
    -o "logs/%J.%I.out" \
    -e "logs/%J.%I.err" \
    -n 36 -R "span[hosts=1]" \
    -W 600 \
    "python Train_ADC_Network.py True \$LSB_JOBINDEX $dataset_name $dimension_init $Time_min 32768 False False True 1 $nan_to" \
    | grep -oE '[0-9]+')
PARALLEL_JOBS+=" && done($ID_C)"

echo "Step 2: Submitted Parallel Jobs B and C (Waiting on A)"

# =================================================================
# STEP 3: Program D (Merge)
# =================================================================
JOB_D_ID=$(bsub \
    -J "CardEst_Merge" \
    -w "$PARALLEL_JOBS" \
    -o "logs/%J.out" \
    -e "logs/%J.err" \
    -n 1 \
    -W 60 \
    "python Train_ADC_Mergetrainset.py $dataset_name"\
    | grep -oE '[0-9]+')

echo "Step 3: Submitted Job D (Waiting on B and C)"

# =================================================================
# STEP 4: Program E (Final Training)
# =================================================================
bsub \
    -J "CardEst_TrainModel" \
    -w "done($JOB_D_ID)" \
    -o "logs/E.out" \
    -e "logs/E.err" \
    -n 36 -R "span[hosts=1]" \
    -W 1200 \
    "python Train_ADC_Network.py False 1 $dataset_name $dimension_init $Time_min 32768 False True False $added_layers $nan_to" > /dev/null

echo "Step 4: Submitted Final Job E (Waiting on D)"
echo "--- All jobs submitted successfully! ---"
