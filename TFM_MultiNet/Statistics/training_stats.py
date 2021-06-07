from TFM.Statistics.StatsModel import TrainingStats

if __name__ == '__main__':
    # Variables
    model_name = "MNet_0"
    id_copy = "_yuv"
    specific_weights = "synthetic_real"

    ts = TrainingStats(model_name+id_copy, specific_weights)

    ts.print_data(y_lim_epoch=[99.6, 99.99], x_lim_loss=[0, 25], title=model_name+id_copy)
