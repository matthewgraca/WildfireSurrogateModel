#!/bin/bash
#experiment_dir="/mnt/wildfire/surrogate-model/2026-08-21-fat-quantized"
#samples=100

while getopts "d:s:c:" opt; do
    case $opt in
        d) EXPERIMENT_DIR="$OPTARG" ;;
        s) SAMPLES="$OPTARG" ;;
        c) CONFIG="$OPTARG" ;;
        *) echo "Usage: $0 [-d experiment directory] [-s samples] [-c config]"; exit 1 ;;
    esac
done

if [[ -z "$EXPERIMENT_DIR" || -z "$SAMPLES" || -z "$CONFIG" ]]; then
    echo "Usage: $0 [-d experiment directory] [-s samples] [-c config]"
    exit 1
fi

echo "Experiment directory set to: $EXPERIMENT_DIR"
echo "Number of samples set to: $SAMPLES"
echo "Config file: $CONFIG"

python scripts/run.py --output-dir $EXPERIMENT_DIR --config $CONFIG && \
python scripts/evaluate.py $EXPERIMENT_DIR --num_samples $SAMPLES --config $CONFIG && \
python scripts/ablation.py $EXPERIMENT_DIR --num_samples $SAMPLES --config $CONFIG
