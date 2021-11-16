from Statistics.StatsModel import TrainingStats

"""
This script generates the metrics of the training process
"""

if __name__ == '__main__':
    # Variables
    model_name = "ViT"
    id_copy = "_cropped_v3_all_512x512"
    specific_weights = "_cropped_v3_all_512x512_epoch_196"

    ts = TrainingStats(model_name+id_copy, specific_weights)

    ts.print_data(y_lim_epoch=[50, 99.99], x_lim_loss=[0, 400], title=model_name+id_copy)
