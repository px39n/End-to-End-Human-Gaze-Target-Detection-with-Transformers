# =============================================================================
# VideoAttTarget dataset dir config
# =============================================================================
videoattentiontarget_data = "D:\\imgData\\videoCo\\images"
videoattentiontarget_train_label = "D:\\imgData\\videoCo\\annotations\\train"
videoattentiontarget_val_label = "D:\\imgData\\videoCo\\annotations\\test"

maxlabeledbboxs=8+2

# =============================================================================
# model config
# =============================================================================
input_resolution = 224
output_resolution = 64

cone_mode = 'early'    # {'late', 'early'} fusion of person information
modality_dropout = True    # only used for attention model
pred_inout = True    # {set True for VideoAttentionTarget}
privacy = False     # {set True to train/test privacy-sensitive model}

# pytorch amp to speed up training and reduce memory usage
use_amp = False