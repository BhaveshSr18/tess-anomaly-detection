import os
import json
import yaml
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path


class RunSaver:
    """Handles saving model runs with configuration, model weights, and results."""
    
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def create_run_directory(self, timestamp=None):
        """Create a timestamped run directory.
        
        Args:
            timestamp (str, optional): Custom timestamp. If None, uses current datetime.
                                      Format: YYYY_MM_DD or YYYY_MM_DD_HH_MM_SS
        
        Returns:
            Path: Path to the created run directory
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        run_dir = self.results_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        return run_dir
    
    def save_run(self, model, config, anomaly_scores, run_dir=None, timestamp=None):
        """Save a complete model run with all components.
        
        Args:
            model (torch.nn.Module): The trained model to save
            config (dict): Configuration dictionary
            anomaly_scores (np.ndarray or pd.DataFrame): Anomaly scores
            run_dir (Path, optional): Existing run directory. If None, creates one with timestamp.
            timestamp (str, optional): Custom timestamp for directory name
        
        Returns:
            Path: Path to the run directory
        """
        if run_dir is None:
            run_dir = self.create_run_directory(timestamp)
        else:
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_path = run_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Save configuration
        config_path = run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✓ Configuration saved to {config_path}")
        
        # Save anomaly scores
        scores_path = run_dir / "anomaly_scores.csv"
        if isinstance(anomaly_scores, pd.DataFrame):
            anomaly_scores.to_csv(scores_path, index=False)
        else:
            # Assume it's numpy array
            pd.DataFrame({"anomaly_score": anomaly_scores}).to_csv(scores_path, index=False)
        print(f"✓ Anomaly scores saved to {scores_path}")
        
        # Save run metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "model_class": model.__class__.__name__,
            "config": config
        }
        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {metadata_path}")
        
        return run_dir
    
    def save_plot(self, run_dir, filename, fig=None):
        """Save a matplotlib figure to the run directory.
        
        Args:
            run_dir (Path): Path to the run directory
            filename (str): Filename for the plot (e.g., 'anomaly_histogram.png')
            fig (matplotlib.figure.Figure, optional): Figure to save. If None, saves current figure.
        
        Returns:
            Path: Path to the saved plot
        """
        run_dir = Path(run_dir)
        plot_path = run_dir / filename
        
        if fig is None:
            import matplotlib.pyplot as plt
            fig = plt.gcf()
        
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        return plot_path
