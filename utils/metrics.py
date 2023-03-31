from typing import Dict, List, Tuple

import torch

def get_metrics_binarybias_cl(ground_truth, epoch_pred, epoch_bias):
    
    #correct instances per class and bias
    correct_1_bias_0 = [1 for i in range(len(epoch_pred)) if epoch_pred[i] == ground_truth[i] and epoch_bias[i] == 0 and epoch_pred[i] == 1]
    correct_1_bias_1 = [1 for i in range(len(epoch_pred)) if epoch_pred[i] == ground_truth[i] and epoch_bias[i] == 1 and epoch_pred[i] == 1]
    correct_0_bias_0 = [1 for i in range(len(epoch_pred)) if epoch_pred[i] == ground_truth[i] and epoch_bias[i] == 0 and epoch_pred[i] == 0]
    correct_0_bias_1 = [1 for i in range(len(epoch_pred)) if epoch_pred[i] == ground_truth[i] and epoch_bias[i] == 1 and epoch_pred[i] == 0]

    #total instances per class and bias
    total_1_bias_0 = [1 for i in range(len(epoch_pred)) if epoch_bias[i] == 0 and ground_truth[i] == 1]
    total_1_bias_1 = [1 for i in range(len(epoch_pred)) if epoch_bias[i] == 1 and ground_truth[i] == 1]
    total_0_bias_0 = [1 for i in range(len(epoch_pred)) if epoch_bias[i] == 0 and ground_truth[i] == 0]
    total_0_bias_1 = [1 for i in range(len(epoch_pred)) if epoch_bias[i] == 1 and ground_truth[i] == 0]

    #getting accuracy and acc per class
    acc = (sum(correct_1_bias_0) + sum(correct_1_bias_1) + sum(correct_0_bias_0) + sum(correct_0_bias_1))/(len(epoch_pred))
    acc_targ_1_bias_0 = sum(correct_1_bias_0) / sum(total_1_bias_0)
    acc_targ_1_bias_1 = sum(correct_1_bias_1) / sum(total_1_bias_1)
    acc_targ_0_bias_0 = sum(correct_0_bias_0) / sum(total_0_bias_0)
    acc_targ_0_bias_1 = sum(correct_0_bias_1) / sum(total_0_bias_1)

    #creating metrics dict
    metrics = {
        "Acc": acc,
        "Acc_T1_B0": acc_targ_1_bias_0,
        "Acc_T1_B1": acc_targ_1_bias_1,
        "Acc_T0_B0": acc_targ_0_bias_0,
        "Acc_T0_B1": acc_targ_0_bias_1
    }
    return metrics

def get_metrics_racebias_cl(epoch_pred, epoch_bias):
    
    correct_white = [1 for i in range(len(epoch_pred)) if epoch_pred[i] == 1 and epoch_bias[i] == 0]
    correct_black = [1 for i in range(len(epoch_pred)) if epoch_pred[i] == 1 and epoch_bias[i] == 1]
    correct_asian = [1 for i in range(len(epoch_pred)) if epoch_pred[i] == 1 and epoch_bias[i] == 2]
    correct_indian = [1 for i in range(len(epoch_pred)) if epoch_pred[i] == 1 and epoch_bias[i] == 3]
    correct_others = [1 for i in range(len(epoch_pred)) if epoch_pred[i] == 1 and epoch_bias[i] == 4]

    #getting accuracy and acc per class
    acc = sum(epoch_pred) / len(epoch_pred)
    acc_white = sum(correct_white) / epoch_bias.count(0)
    acc_black = sum(correct_black) / epoch_bias.count(1)
    acc_asian = sum(correct_asian) / epoch_bias.count(2)
    acc_indian = sum(correct_indian) / epoch_bias.count(3)
    acc_others = sum(correct_others) / epoch_bias.count(4)

    #creating metrics dict
    metrics = {
        "Acc": acc,
        "Acc_White": acc_white,
        "Acc_Black": acc_black,
        "Acc_Asian": acc_asian,
        "Acc_Indian": acc_indian,
        "Acc_Others": acc_others
    }
    return metrics

def define_weight(acc: float, n_classes: int):
    random_acc = 1/n_classes
    norm_value = (acc - random_acc) / (1 - random_acc)
    weight = 1 - norm_value
    return weight

if __name__ == "__main__":
    # epoch_pred = [1,0,1,0,1,1,1,1,0,0,0,0,1,1]
    # epoch_bias = [0,0,0,0,0,0,0,0,1,1,1,1,1,1]
    # print('male list', sum(epoch_bias))
    # print('female list', len(epoch_bias) - sum(epoch_bias))
    # print('correct list', sum(epoch_pred))
    # print('n_samples', len(epoch_pred))
    # print('acc', sum(epoch_pred) / len(epoch_pred))

    # get_metrics_genderbias_cl(epoch_pred, epoch_bias)

    epoch_pred = [1,0,1,0,1,1,1,1,0,0,0,0,1,1]
    epoch_bias = [0,0,1,1,2,2,3,3,4,4,0,0,1,1]
    #counting number of samples per class
    print('white list', epoch_bias.count(0))
    print('black list', epoch_bias.count(1))
    print('asian list', epoch_bias.count(2))
    print('indian list', epoch_bias.count(3))
    print('others list', epoch_bias.count(4))

    print('correct list', sum(epoch_pred))
    print('n_samples', len(epoch_pred))
    print('acc', sum(epoch_pred) / len(epoch_pred))   
    metrics=get_metrics_racebias_cl(epoch_pred, epoch_bias)
    print(metrics)
    
