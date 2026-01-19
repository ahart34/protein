import pandas as pd
import matplotlib.pyplot as plt
import os

# Models and tasks
MODELS = ['bert', 'esm', 'albert']
TASKS = ['cl', 'ec', 'go']

# Task display names
TASK_NAMES = {
    'cl': 'CL',
    'ec': 'EC',
    'go': 'GO'
}

# Colors for different tasks
TASK_COLORS = {
    'cl': 'orange',
    'ec': 'blue',
    'go': 'green'
}

# Line styles and markers for different models
MODEL_STYLES = {
    'bert': {'linestyle': '-', 'marker': 'o', 'label_suffix': 'BERT'},
    'esm': {'linestyle': '--', 'marker': 's', 'label_suffix': 'ESM'},
    'albert': {'linestyle': ':', 'marker': '^', 'label_suffix': 'ALBERT'}
}

# Create output directory if it doesn't exist
os.makedirs('results/plots', exist_ok=True)

def load_confidence_data(model, task):
    """Load confidence data for a specific model/task combination."""
    file_path = f'results/confidence/{model}_{task}_confidence.csv'
    
    if not os.path.exists(file_path):
        return None
    
    try:
        df = pd.read_csv(file_path)
        # Check if required columns exist
        if 'layer' in df.columns and 'excess_aurc' in df.columns:
            # Sort by layer
            df = df.sort_values('layer')
            return df
        else:
            print(f"Missing required columns in {file_path}")
            return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_confidence_metrics():
    """Create confidence plot with all model/task combinations."""
    print("Creating confidence plot...")
    
    # Create plot
    plt.figure(figsize=(10, 7))
    
    # Plot data for each model and task combination
    for task in TASKS:
        task_label = TASK_NAMES.get(task, task.upper())
        color = TASK_COLORS.get(task, 'gray')
        
        for model in MODELS:
            df = load_confidence_data(model, task)
            
            if df is not None:
                style = MODEL_STYLES[model]
                label = f"{task_label} ({style['label_suffix']})"
                
                plt.plot(df['layer'], df['excess_aurc'],
                        color=color,
                        linestyle=style['linestyle'],
                        marker=style['marker'],
                        markersize=4,
                        linewidth=2,
                        label=label,
                        alpha=0.8)
                print(f"Plotted {model}_{task}")
            else:
                print(f"Skipping {model}_{task} (file not found or invalid)")
    
    # Set labels and title
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Excess AURC', fontsize=14)
    plt.title('Confidence Calibration: Excess AURC by Layer', fontsize=16)
    
    # Legend
    plt.legend(loc='best', fontsize=10, framealpha=0.9, ncol=1)
    
    # Grid
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_file = 'results/plots/confidence_excess_aurc.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved confidence plot to {output_file}")

# Main execution
if __name__ == "__main__":
    plot_confidence_metrics()
    print("\nConfidence plot generated successfully!")

