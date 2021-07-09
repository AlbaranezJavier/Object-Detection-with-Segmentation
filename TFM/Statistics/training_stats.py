from TFM.Statistics.StatsModel import TrainingStats

if __name__ == '__main__':
    # Variables
    model_name = "Net_5"
    id_copy = "_cls_yuv"
    specific_weights = "synthetic_real_cls_yuv_epoch_123"

    ts = TrainingStats(model_name+id_copy, specific_weights)

    ts.print_data(y_lim_epoch=[99.6, 99.99], x_lim_loss=[0, 25], title=model_name+id_copy)
