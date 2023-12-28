from glob import glob, iglob
import matplotlib.pyplot as plt
import pandas as pd
import os


def get_w5_curves(baseline = 'baseline_1', w5_list=['1', '2', '3','5', '7', '10', '12']):
    path = './results/' + baseline +'/'+ baseline 
    acc_list, pre_list, rec_list, f1_list = [], [], [], []
    for w5 in w5_list:
        prefix = path + '_W5_'+ w5
        df = pd.read_csv(prefix+'.csv')
        acc_list.append((int(w5), df.accuracy.mean()))
        pre_list.append((int(w5), df.precision.mean()))
        rec_list.append((int(w5), df.recall.mean()))
        f1_list.append((int(w5), df.f1_score.mean()))
                       

    # figure(figsize=(16, 12), dpi=80)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].plot(*zip(*acc_list), linewidth=2)
    axs[0, 0].set_title('Accuracy')
    axs[0, 1].plot(*zip(*pre_list),linewidth=2)
    axs[0, 1].set_title('Precision')
    axs[1, 0].plot(*zip(*rec_list),linewidth=2) 
    axs[1, 0].set_title('Recall')
    axs[1, 1].plot(*zip(*f1_list), linewidth=2) 
    axs[1, 1].set_title('F1 Score')

    for ax in axs.flat:
        ax.set(xlabel='w5', ylabel='score')
    fig.suptitle(baseline)
