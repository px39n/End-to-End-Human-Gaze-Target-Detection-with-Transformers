
from config import videoattentiontarget_train_label,videoattentiontarget_val_label
import pandas as pd
import glob
import os
def process( annotation_dir,train=True ):

    shows = glob.glob(os.path.join(annotation_dir, '*'))

    df = pd.DataFrame()
    max_label=0
    for s in shows:
        sequence_annotations = glob.glob(os.path.join(s, '*', '*.txt'))
        video_label=dict()
        for ann in sequence_annotations:
            parts = ann.split(os.path.sep)
            show = parts[-3]
            clip = parts[-2]

            df_tmp = pd.read_csv(ann, header=None, index_col=False,
                                 names=['path', 'xmin', 'ymin', 'xmax', 'ymax', 'gazex', 'gazey'])
            df_tmp['path'] = [os.path.join(show, clip, p) for p in df_tmp['path'].values]
            video_tmp=df_tmp.to_dict()
            tmp={video_tmp["path"][i]:{key:video_tmp[key][i] for key in [ 'xmin', 'ymin', 'xmax', 'ymax', 'gazex', 'gazey']} for i in range(df_tmp.shape[0])}

            video_label=merge_data(video_label,tmp)
        max_label=max(max_label,max([len(value) for value in video_label.values()])if video_label else 0)
        video_=split_data(video_label,split_size=5,data_key=show)
        df_= pd.DataFrame(video_)

        df = pd.concat([df, df_])

    if train:
        file_name="train.csv"
    else :
        file_name="test.csv"
    print(f"print_max_labeled:{max_label}")
    df.to_csv(os.path.join(annotation_dir,file_name))

def split_data(video_label,split_size,data_key):
    ret=[]
    one_sp=[]
    s=0
    for key,label in video_label.items():

        if s==split_size:
            s=0
            ret.append({"data":one_sp})
            one_sp=[]
        one_sp.append({"path":key,"label":label})
        s += 1
    return ret


def merge_data(a,b):
    for i in b.keys():
        if i in a:
            a[i].append(b[i])
        else:
            a[i]=[b[i]]
    return a

if __name__=="__main__":
    process(videoattentiontarget_train_label)
    process(videoattentiontarget_val_label)