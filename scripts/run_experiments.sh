#!/bin/bash
# =============================================================================
# SolidCoder Experiment Runner
# Reproduces experiments from the paper
# =============================================================================

set -e  # Exit on error

# ============================================
# CONFIGURATION
# ============================================

# Check API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY environment variable is not set"
    echo "Usage: export OPENROUTER_API_KEY='your-api-key'"
    exit 1
fi

# Change to project root
cd "$(dirname "$0")/.."

# Models used in paper (OpenRouter IDs)
MODELS=("openai/gpt-4o-2024-08-06" "openai/gpt-oss-120b" "x-ai/grok-4.1-fast")

# All strategies
STRATEGIES=("Direct" "CoT" "SelfPlanning" "Analogical" "MapCoder" "CodeSIM" "SolidCoder")

# All datasets
DATASETS=("HumanEval" "CC" "APPS")

# ============================================
# HELPER FUNCTIONS
# ============================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_experiment() {
    local model=$1
    local strategy=$2
    local dataset=$3

    log "Starting: $strategy / $model / $dataset"
    
    local cmd="PYTHONPATH=./src python src/main.py"
    cmd+=" --strategy $strategy"
    cmd+=" --dataset $dataset"
    cmd+=" --model $model"
    cmd+=" --model_provider OpenRouter"
    cmd+=" --temperature 0"
    cmd+=" --verbose 1"
    cmd+=" --store_log_in_file yes"
    cmd+=" --cont yes"

    # SolidCoder needs all S.O.L.I.D. component flags
    if [ "$strategy" == "SolidCoder" ]; then
        cmd+=" --enable_shift_left"
        cmd+=" --enable_oracle_assert"
        cmd+=" --enable_live_verify"
        cmd+=" --enable_inter_sim"
        cmd+=" --enable_defensive_test"
    fi

    eval $cmd
    
    log "Completed: $strategy / $model / $dataset"
}

run_ablation() {
    local model=$1
    local ablation=$2  # woS, woO, woL, woI, woD
    
    log "Starting ablation: $model / $ablation"
    
    local cmd="PYTHONPATH=./src python src/main.py"
    cmd+=" --strategy SolidCoder"
    cmd+=" --dataset CC"
    cmd+=" --model $model"
    cmd+=" --model_provider OpenRouter"
    cmd+=" --temperature 0"
    cmd+=" --verbose 1"
    cmd+=" --store_log_in_file yes"
    
    # Enable all except the ablated component
    case $ablation in
        "woS") cmd+=" --enable_oracle_assert --enable_live_verify --enable_inter_sim --enable_defensive_test" ;;
        "woO") cmd+=" --enable_shift_left --enable_live_verify --enable_inter_sim --enable_defensive_test" ;;
        "woL") cmd+=" --enable_shift_left --enable_oracle_assert --enable_inter_sim --enable_defensive_test" ;;
        "woI") cmd+=" --enable_shift_left --enable_oracle_assert --enable_live_verify --enable_defensive_test" ;;
        "woD") cmd+=" --enable_shift_left --enable_oracle_assert --enable_live_verify --enable_inter_sim" ;;
    esac
    
    eval $cmd
    
    log "Completed ablation: $model / $ablation"
}

show_usage() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  single <model> <strategy> <dataset>  Run single experiment"
    echo "  all                                   Run all experiments (63 total)"
    echo "  ablation <model>                      Run ablation study on CC"
    echo "  help                                  Show this message"
    echo ""
    echo "Examples:"
    echo "  $0 single gpt-4o SolidCoder HumanEval"
    echo "  $0 ablation gpt-4o"
    echo "  $0 all"
    echo ""
    echo "Models: ${MODELS[*]}"
    echo "Strategies: ${STRATEGIES[*]}"
    echo "Datasets: ${DATASETS[*]}"
}

# ============================================
# MAIN
# ============================================

case "${1:-help}" in
    single)
        if [ $# -lt 4 ]; then
            echo "Error: single requires <model> <strategy> <dataset>"
            exit 1
        fi
        run_experiment "$2" "$3" "$4"
        ;;
    all)
        log "==========================================="
        log "Running ALL experiments (${#MODELS[@]} × ${#STRATEGIES[@]} × ${#DATASETS[@]} = $((${#MODELS[@]} * ${#STRATEGIES[@]} * ${#DATASETS[@]})) total)"
        log "==========================================="
        for model in "${MODELS[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                for strategy in "${STRATEGIES[@]}"; do
                    run_experiment "$model" "$strategy" "$dataset"
                done
            done
        done
        log "All experiments completed!"
        ;;
    ablation)
        model="${2:-gpt-4o}"
        log "==========================================="
        log "Running ablation study for $model on CC"
        log "==========================================="
        for ablation in woS woO woL woI woD; do
            run_ablation "$model" "$ablation"
        done
        log "Ablation study completed!"
        ;;
    help|*)
        show_usage
        ;;
esac
