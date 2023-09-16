
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
```

\end{verbatim}

---

