from TFM.Perception.Perception import Perception

if __name__ == '__main__':
    # Variables
    original_img_shape = (720, 1280, 3)
    nn_model = "Net_5_reg"
    nn_epoch = 1208
    nn_weights_path = r"D:\Work\Repositorios\JaviProject\TFM\Weights\Net_5\synthetic_real_yuv_r_epoch"
    data_saved_format = dict
    vs_on = True
    ir_path = r"C:\Users\TTe_J\Downloads\SyntheticConeDataset(1005)\RightImages"

    pc = Perception(original_img_shape, nn_model, nn_epoch, nn_weights_path, data_saved_format, vs_on, ir_path)
    pc.run()

q