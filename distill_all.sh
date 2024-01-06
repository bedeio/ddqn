#!/bin/bash

# Define the function
run_distill_model() {
    local env_names=('LunarLander-v2' 'Taxi-v3' 'CartPole-v1')

    for env in "${env_names[@]}"; do
        echo "Running distill_model.py for environment: $env"
        python distill_model.py --env_name "$env"
    done
}

# Call the function
run_distill_model
