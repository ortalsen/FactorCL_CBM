import matplotlib.pyplot as plt
import numpy as np
import wandb

def selected_conf_plot(conf_matrix, selected_classes):

    conf_matrix_values = conf_matrix.compute().detach().cpu().numpy()
    selected_conf_matrix_values = conf_matrix_values[selected_classes][:, selected_classes]

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(selected_conf_matrix_values, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Selected Classes')
    plt.colorbar()

    # Add labels
    tick_marks = np.arange(len(selected_classes))
    plt.xticks(tick_marks, selected_classes)
    plt.yticks(tick_marks, selected_classes)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Add text annotations
    thresh = selected_conf_matrix_values.max() / 2.
    for i in range(len(selected_classes)):
        for j in range(len(selected_classes)):
            plt.text(j, i, format(selected_conf_matrix_values[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if selected_conf_matrix_values[i, j] > thresh else "black")

    plt.tight_layout()
    
    #log plot to wandb
    wandb_image = wandb.Image(plt)
    wandb.log({"final_model/confusion_matrix": wandb_image})