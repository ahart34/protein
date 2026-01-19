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
SKIP_LIST = ['albert_cl']

# Models and tasks
MODELS = ['bert', 'albert', 'esm']
TASKS = ['cl', 'ec', 'go', 'ssp']

# Task display names
TASK_NAMES = {
    'cl': 'CL',
    'ec': 'EC',
    'go': 'GO',
    'ssp': 'SSP'
}

# Colors for different tasks
TASK_COLORS = {
    'cl': 'orange',
    'ec': 'blue',
    'go': 'green',
    'ssp': 'purple'
}

# Line styles and markers for different models
MODEL_STYLES = {
    'bert': {'linestyle': '-', 'marker': 'o', 'label_suffix': 'BERT'},
    'esm': {'linestyle': '--', 'marker': 's', 'label_suffix': 'ESM'},
    'albert': {'linestyle': ':', 'marker': '^', 'label_suffix': 'ALBERT'}
}

# Create output directory if it doesn't exist
os.makedirs('results/plots', exist_ok=True)

def load_data_for_model_task(model, task):
    """Load eval and exit_max data for a specific model/task combination."""
    model_task = f"{model}_{task}"
    
    # Skip if in skip list
    if model_task in SKIP_LIST:
        return None
    
    # Check if files exist
    if model_task not in FILE_MAPPING:
        return None
    
    files = FILE_MAPPING[model_task]
    data = {}
    
    # Load eval file
    if 'eval' in files and os.path.exists(files['eval']):
        try:
            df = pd.read_csv(files['eval'])
            # Check if required columns exist
            if 'Layer' in df.columns and 'Time' in df.columns:
                # Sort by Layer
                df = df.sort_values('Layer')
                data['eval'] = df
        except Exception as e:
            print(f"Error loading {files['eval']}: {e}")
    
    # Load exit_max file
    if 'exit_max' in files and os.path.exists(files['exit_max']):
        try:
            df = pd.read_csv(files['exit_max'])
            # Check if required columns exist
            if 'Average Computed Layer' in df.columns and 'Time' in df.columns:
                # Sort by Average Computed Layer
                df = df.sort_values('Average Computed Layer')
                data['exit_max'] = df
        except Exception as e:
            print(f"Error loading {files['exit_max']}: {e}")
    
    return data if data else None

def plot_all_walltimes():
    """Create combined walltime plot for all model/task combinations."""
    print("Creating combined walltime plot...")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot data for each model and task combination
    for task in TASKS:
        task_label = TASK_NAMES.get(task, task.upper())
        color = TASK_COLORS.get(task, 'gray')
        
        for model in MODELS:
            data = load_data_for_model_task(model, task)
            
            if data:
                style = MODEL_STYLES.get(model, {'linestyle': '-', 'marker': 'o', 'label_suffix': model.upper()})
                
                # Plot exit_max line (Early Exit - Max)
                if 'exit_max' in data:
                    df = data['exit_max']
                    label = f"{task_label} ({style['label_suffix']}) Early Exit - Max"
                    plt.plot(df['Average Computed Layer'], df['Time'],
                            color=color,
                            linestyle=style['linestyle'],
                            marker=style['marker'],
                            markersize=4,
                            linewidth=2,
                            label=label,
                            alpha=0.8)
                    print(f"Plotted exit_max for {model}_{task}")
                
                # Plot diamond at (last_layer, last_layer_walltime) from eval
                if 'eval' in data:
                    df = data['eval']
                    # Get the last layer (max Layer value) and its walltime
                    last_row = df.loc[df['Layer'].idxmax()]
                    last_layer = last_row['Layer']
                    last_time = last_row['Time']
                    
                    # Plot diamond marker
                    label_diamond = f"{task_label} ({style['label_suffix']}) Last Layer"
                    plt.plot(last_layer, last_time,
                            marker='D',
                            markersize=8,
                            color=color,
                            markeredgecolor='black',
                            markeredgewidth=1.5,
                            linestyle='',
                            label=label_diamond,
                            alpha=0.9)
                    print(f"Plotted last layer diamond for {model}_{task} at ({last_layer}, {last_time:.2f})")
            else:
                print(f"Skipping {model}_{task} (data not available)")
    
    # Set labels and title
    plt.xlabel('Average Computed Layer', fontsize=14)
    plt.ylabel('Walltime (seconds)', fontsize=14)
    plt.title('Walltime vs Average Computed Layer', fontsize=16)
    
    # Legend (outside plot area to avoid clutter)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
    
    # Grid
    plt.grid(True, alpha=0.3)
    
    # Save figure with extra space for legend
    output_file = 'results/plots/all_walltimes.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved combined walltime plot to {output_file}")

# Main execution
if __name__ == "__main__":
    plot_all_walltimes()
    print("\nWalltime plot generated successfully!")

