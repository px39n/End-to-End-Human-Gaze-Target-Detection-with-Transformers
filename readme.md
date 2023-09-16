

**Disclaimer:** This repository is an unofficial implementation of the research paper [End-to-End Human-Gaze-Target Detection with Transformers](https://arxiv.org/abs/2203.10433). Our aim is to promote development in related areas of interest. We are not responsible for any bugs or discrepancies in the results obtained using this codebase. 

**Abstract**
In this paper, we propose an effective and efficient method for Human-Gaze-Target (HGT) detection, i.e., gaze following. Current approaches decouple the HGT detection task into separate branches of salient object detection and human gaze prediction, employing a two-stage framework where human head locations must first be detected and then be fed into the next gaze target prediction sub-network. In contrast, we redefine the HGT detection task as detecting human head locations and their gaze targets, simultaneously. By this way, our method, named Human-Gaze-Target detection TRansformer or HGTTR, streamlines the HGT detection pipeline by eliminating all other additional components. HGTTR reasons about the relations of salient objects and human gaze from the global image context. Moreover, unlike existing two-stage methods that require human head locations as input and can predict only one human's gaze target at a time, HGTTR can directly predict the locations of all people and their gaze targets at one time in an end-to-end manner. The effectiveness and robustness of our proposed method are verified with extensive experiments on the two standard benchmark datasets, GazeFollowing and VideoAttentionTarget. Without bells and whistles, HGTTR outperforms existing state-of-the-art methods by large margins (6.4 mAP gain on GazeFollowing and 10.3 mAP gain on VideoAttentionTarget) with a much simpler architecture.


#### Data Preparation
Download the VideoAttentionTarget dataset from the following [link](https://www.dropbox.com/s/8ep3y1hd74wdjy5/videoattention.target.zip?dl=0). Note that accessing this link may require a VPN.

#### Training on VideoAttentionTarget

### Path Configuration
```plaintext
videoattention_target_data = "D:\\imgData\\videoCo\\images"  % Path to the images
videoattentiontarget_train_label = "D:\\imgData\\videoCo\\annotations\\train" % Path to the training annotations
videoattentiontarget_val_label = "D:\\imgData\\videoCo\\annotations\\test" % Path to the testing annotations
maxlabeledbboxs = 8 + 2  % Maximum number of objects to predict, 8+2 implies 8 in training set plus 2 additional during prediction
```

#### Data Preprocessing
```plaintext
python data_opt/pre_process_data.py  % Preprocess the annotation data
% During preprocessing, adjacent frames are grouped as similar images. During training, a random frame from these is selected.
```

### Training
```plaintext
-- heat_dim  [6,6,40]  % [w0,h0,gaussian] Pixel and value dimensions
-- backbone  resnet50  % Choices: [resnet50, resnet101]
-- lr_backbone  % Learning rate for the backbone
-- lr  % General learning rate
-- output_dir  % Directory to save the model
-- batch_size
-- epochs
python train_on_videoattarget.py  % Start training
```

### File Structure
```plaintext
- models
    matcher.py  % Hungarian Algorithm for matching predictions and ground truth to avoid overfitting
    detr.py  % The DETR model
        SetCriterion  % Computes loss
    resnet.py  % Backbone network. Pretrained config for pre-trained resnet model (requires a proxy if set to True)
- data_opt
    heat_map.py  % Generates heatmaps based on annotated attention coordinates
    pre_process_data.py  % Data preprocessing
    load_data.py  % Data loading class
```

### Prediction
```plaintext
-- checkpoint_path  % Path to the saved model
-- prediction_data  % Path to the image data
python predict.py
```

Example Output:
```plaintext
{
  "box":[x_min,y_min,x_max,y_max],
  "watchinside":1,  % 1 indicates that the gaze is inside the box
  "heatmap":w_dim*h_dim
}

