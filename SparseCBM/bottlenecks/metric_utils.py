from cbm import *
from configs import *
from trainer_utils import *
from typing import List, Dict, Optional

def plot_trainer_metrics(hist: List[Dict]):
    """
    Function which plots metrics in metrics_to_draw dict for trainer.hist
    Example of usage:
        trainer.train()
        plot_trainer_metrics(trainer.hist)
    """
    num_rows = 3
    num_cols = 4
    for metrics_dict in hist:
        print_centered_text(metrics_dict["name"])
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        metrics_to_draw = [
       'train_loss', 'train_cbl_loss', 
       'train_acc_top_1', 'train_acc_top_5',
       'val_loss', 'val_cbl_loss',
       'val_acc_top_1', 'val_acc_top_5',
       'val_precision', 'val_recall', 'val_f1', 
       'train_x']
        drawn_metrics_dict = {k: metrics_dict[k] for k in metrics_to_draw if k in metrics_dict}
        for idx, (metric_name, metric_values) in enumerate(drawn_metrics_dict.items()):
            if metric_name in metrics_to_draw:
                row_idx = idx // num_cols
                col_idx = idx % num_cols
                
                ax = axs[row_idx, col_idx] if len(metrics_dict.keys()) > 1 else axs
                ax.plot(metric_values)
                ax.set_title(metric_name)
                
        plt.tight_layout()
        plt.grid(True)
        plt.show()


def draw_confusion_matrix(model, loader):
    pass


def interpretability(image, model, topk=(1,)):
    pass
