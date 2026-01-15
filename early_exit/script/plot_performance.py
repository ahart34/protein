import pandas as pd
import matplotlib.pyplot as plt
import os

# Hardcoded dictionary mapping models and datasets to output files
FILE_MAPPING = {
    'bert_cl': {
        'eval': 'results/metrics/bert_cl_eval.csv',
        'exit_last': 'results/metrics/bert_cl_exit_last.csv',
        'exit_max': 'results/metrics/bert_cl_exit_max.csv'
    },
    'bert_ec': {
        'eval': 'results/metrics/bert_ec_eval.csv',
        'exit_last': 'results/metrics/bert_ec_exit_last.csv',
        'exit_max': 'results/metrics/bert_ec_exit_max.csv'
    },
    'bert_go': {
        'eval': 'results/metrics/bert_go_eval.csv',
        'exit_last': 'results/metrics/bert_go_exit_last.csv',
        'exit_max': 'results/metrics/bert_go_exit_max.csv'
    },
    'bert_ssp': {
        'eval': 'results/metrics/bert_ssp_eval.csv',
        'exit_last': 'results/metrics/bert_ssp_exit_last.csv',
        'exit_max': 'results/metrics/bert_ssp_exit_max.csv'
    },
    'albert_cl': {
        'eval': 'results/metrics/albert_cl_eval.csv',
        'exit_last': 'results/metrics/albert_cl_exit_last.csv',
        'exit_max': 'results/metrics/albert_cl_exit_max.csv'
    },
    'albert_ec': {
        'eval': 'results/metrics/albert_ec_eval.csv',
        'exit_last': 'results/metrics/albert_ec_exit_last.csv',
        'exit_max': 'results/metrics/albert_ec_exit_max.csv'
    },
    'albert_go': {
        'eval': 'results/metrics/albert_go_eval.csv',
        'exit_last': 'results/metrics/albert_go_exit_last.csv',
        'exit_max': 'results/metrics/albert_go_exit_max.csv'
    },
    'albert_ssp': {
        'eval': 'results/metrics/albert_ssp_eval.csv',
        'exit_last': 'results/metrics/albert_ssp_exit_last.csv',
        'exit_max': 'results/metrics/albert_ssp_exit_max.csv'
    },
    'esm_cl': {
        'eval': 'results/metrics/esm_cl_eval.csv',
        'exit_last': 'results/metrics/esm_cl_exit_last.csv',
        'exit_max': 'results/metrics/esm_cl_exit_max.csv'
    },
    'esm_ec': {
        'eval': 'results/metrics/esm_ec_eval.csv',
        'exit_last': 'results/metrics/esm_ec_exit_last.csv',
        'exit_max': 'results/metrics/esm_ec_exit_max.csv'
    },
    'esm_go': {
        'eval': 'results/metrics/esm_go_eval.csv',
        'exit_last': 'results/metrics/esm_go_exit_last.csv',
        'exit_max': 'results/metrics/esm_go_exit_max.csv'
    },
    'esm_ssp': {
        'eval': 'results/metrics/esm_ssp_eval.csv',
        'exit_last': 'results/metrics/esm_ssp_exit_last.csv',
        'exit_max': 'results/metrics/esm_ssp_exit_max.csv'
    }
}

# List of model/task combinations to skip
SKIP_LIST = ['bert_cl', 'esm_go', 'esm_ssp']

# Models and tasks
MODELS = ['bert', 'albert', 'esm']
TASKS = ['cl', 'ec', 'go', 'ssp']

# Create output directory if it doesn't exist
os.makedirs('results/plots', exist_ok=True)

def plot_model_task(model, task):
    """Create performance plot for a specific model/task combination."""
    model_task = f"{model}_{task}"
    
    # Skip if in skip list
    if model_task in SKIP_LIST:
        print(f"Skipping {model_task} (in skip list)")
        return
    
    # Check if files exist
    if model_task not in FILE_MAPPING:
        print(f"missing data for {model_task}")
        return
    
    files = FILE_MAPPING[model_task]
    
    # Check if all required files exist
    missing_files = []
    for file_type, file_path in files.items():
        if not os.path.exists(file_path):
            missing_files.append(file_type)
    
    if missing_files:
        print(f"missing data for {model_task}")
        return
    
    # Load data
    try:
        eval_df = pd.read_csv(files['eval'])
        exit_last_df = pd.read_csv(files['exit_last'])
        exit_max_df = pd.read_csv(files['exit_max'])
    except Exception as e:
        print(f"missing data for {model_task}")
        return
    
    # Determine which performance metric to use
    perf_metric = None
    for metric in ['f1_max', 'acc', 'macro_acc']:
        if metric in eval_df.columns:
            perf_metric = metric
            break
    
    if perf_metric is None:
        print(f"missing data for {model_task} (no performance metric found)")
        return
    
    # Sort dataframes by Average Computed Layer to ensure proper line plotting
    exit_last_df = exit_last_df.sort_values('Average Computed Layer')
    exit_max_df = exit_max_df.sort_values('Average Computed Layer')
    
    # Create plot
    plt.figure(figsize=(8, 6))
    
    # Plot eval data (Layer vs Performance) - Blue
    plt.plot(eval_df['Layer'], eval_df[perf_metric], marker='o', markersize=4, linewidth=2, color='blue')
    
    # Plot early exit last (Average Computed Layer vs Performance) - Green
    plt.plot(exit_last_df['Average Computed Layer'], exit_last_df[perf_metric], 
             marker='s', markersize=4, linewidth=2, color='green')
    
    # Plot early exit max (Average Computed Layer vs Performance) - Orange
    plt.plot(exit_max_df['Average Computed Layer'], exit_max_df[perf_metric], 
             marker='^', markersize=4, linewidth=2, color='orange')
    
    # Draw horizontal line for last layer performance
    last_layer_perf = eval_df[perf_metric].iloc[-1]
    plt.axhline(y=last_layer_perf, color='black', linestyle='-', linewidth=2)
    
    # Set labels and title
    plt.xlabel('Average Number of Computed Layers', fontsize=14)
    plt.ylabel('Performance', fontsize=14)
    plt.title(f'{task.upper()} - {model.upper()}', fontsize=18)
    
    # Grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_file = f'results/plots/{model_task}.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot for {model_task} to {output_file}")

# Main execution
if __name__ == "__main__":
    for model in MODELS:
        for task in TASKS:
            plot_model_task(model, task)

