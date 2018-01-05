https://github.com/zsdonghao/u-net-brain-tumor

[toc]

# prepare_data_with_valid.py

```python
import tensorlayer as tl
import numpy as np
import os, csv, random, gc, pickle
import nibabel as nib


"""
In seg file
--------------
Label 1: necrotic and non-enhancing tumor
Label 2: edema 
Label 4: enhancing tumor
Label 0: background
MRI
-------
whole/complete tumor: 1 2 4
core: 1 4
enhance: 4
"""
###============================= SETTINGS ===================================###
DATA_SIZE = 'half' # (small, half or all)

save_dir = "data/train_dev_all/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

HGG_data_path = "data/Brats17TrainingData/HGG"
LGG_data_path = "data/Brats17TrainingData/LGG"
survival_csv_path = "data/Brats17TrainingData/survival_data.csv"
###==========================================================================###

survival_id_list = []
survival_age_list =[]
survival_peroid_list = []

with open(survival_csv_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for idx, content in enumerate(reader):
        survival_id_list.append(content[0])
        survival_age_list.append(float(content[1]))
        survival_peroid_list.append(float(content[2]))

print(len(survival_id_list)) #163

if DATA_SIZE == 'all':
    HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)
    LGG_path_list = tl.files.load_folder_list(path=LGG_data_path)
elif DATA_SIZE == 'half':
    HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)[0:100]# DEBUG WITH SMALL DATA
    LGG_path_list = tl.files.load_folder_list(path=LGG_data_path)[0:30] # DEBUG WITH SMALL DATA
elif DATA_SIZE == 'small':
    HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)[0:50] # DEBUG WITH SMALL DATA
    LGG_path_list = tl.files.load_folder_list(path=LGG_data_path)[0:20] # DEBUG WITH SMALL DATA
else:
    exit("Unknow DATA_SIZE")
print(len(HGG_path_list), len(LGG_path_list)) #210 #75

HGG_name_list = [os.path.basename(p) for p in HGG_path_list]
LGG_name_list = [os.path.basename(p) for p in LGG_path_list]

survival_id_from_HGG = []
survival_id_from_LGG = []
for i in survival_id_list:
    if i in HGG_name_list:
        survival_id_from_HGG.append(i)
    elif i in LGG_name_list:
        survival_id_from_LGG.append(i)
    else:
        print(i)

print(len(survival_id_from_HGG), len(survival_id_from_LGG)) #163, 0

# use 42 from 210 (in 163 subset) and 15 from 75 as 0.8/0.2 train/dev split

# use 126/42/42 from 210 (in 163 subset) and 45/15/15 from 75 as 0.6/0.2/0.2 train/dev/test split
index_HGG = list(range(0, len(survival_id_from_HGG)))
index_LGG = list(range(0, len(LGG_name_list)))
# random.shuffle(index_HGG)
# random.shuffle(index_HGG)

if DATA_SIZE == 'all':
    dev_index_HGG = index_HGG[-84:-42]
    test_index_HGG = index_HGG[-42:]
    tr_index_HGG = index_HGG[:-84]
    dev_index_LGG = index_LGG[-30:-15]
    test_index_LGG = index_LGG[-15:]
    tr_index_LGG = index_LGG[:-30]
elif DATA_SIZE == 'half':
    dev_index_HGG = index_HGG[-30:]  # DEBUG WITH SMALL DATA
    test_index_HGG = index_HGG[-5:]
    tr_index_HGG = index_HGG[:-30]
    dev_index_LGG = index_LGG[-10:]  # DEBUG WITH SMALL DATA
    test_index_LGG = index_LGG[-5:]
    tr_index_LGG = index_LGG[:-10]
elif DATA_SIZE == 'small':
    dev_index_HGG = index_HGG[35:42]   # DEBUG WITH SMALL DATA
    # print(index_HGG, dev_index_HGG)
    # exit()
    test_index_HGG = index_HGG[41:42]
    tr_index_HGG = index_HGG[0:35]
    dev_index_LGG = index_LGG[7:10]    # DEBUG WITH SMALL DATA
    test_index_LGG = index_LGG[9:10]
    tr_index_LGG = index_LGG[0:7]

survival_id_dev_HGG = [survival_id_from_HGG[i] for i in dev_index_HGG]
survival_id_test_HGG = [survival_id_from_HGG[i] for i in test_index_HGG]
survival_id_tr_HGG = [survival_id_from_HGG[i] for i in tr_index_HGG]

survival_id_dev_LGG = [LGG_name_list[i] for i in dev_index_LGG]
survival_id_test_LGG = [LGG_name_list[i] for i in test_index_LGG]
survival_id_tr_LGG = [LGG_name_list[i] for i in tr_index_LGG]

survival_age_dev = [survival_age_list[survival_id_list.index(i)] for i in survival_id_dev_HGG]
survival_age_test = [survival_age_list[survival_id_list.index(i)] for i in survival_id_test_HGG]
survival_age_tr = [survival_age_list[survival_id_list.index(i)] for i in survival_id_tr_HGG]

survival_period_dev = [survival_peroid_list[survival_id_list.index(i)] for i in survival_id_dev_HGG]
survival_period_test = [survival_peroid_list[survival_id_list.index(i)] for i in survival_id_test_HGG]
survival_period_tr = [survival_peroid_list[survival_id_list.index(i)] for i in survival_id_tr_HGG]

data_types = ['flair', 't1', 't1ce', 't2']
data_types_mean_std_dict = {i: {'mean': 0.0, 'std': 1.0} for i in data_types}

# calculate mean and std for all data types

# preserving_ratio = 0.0
# preserving_ratio = 0.01 # 0.118 removed
# preserving_ratio = 0.05 # 0.213 removed
# preserving_ratio = 0.10 # 0.359 removed

#==================== LOAD ALL IMAGES' PATH AND COMPUTE MEAN/ STD
for i in data_types:
    data_temp_list = []
    for j in HGG_name_list:
        img_path = os.path.join(HGG_data_path, j, j + '_' + i + '.nii.gz')
        img = nib.load(img_path).get_data()
        data_temp_list.append(img)

    for j in LGG_name_list:
        img_path = os.path.join(LGG_data_path, j, j + '_' + i + '.nii.gz')
        img = nib.load(img_path).get_data()
        data_temp_list.append(img)

    data_temp_list = np.asarray(data_temp_list)
    m = np.mean(data_temp_list)
    s = np.std(data_temp_list)
    data_types_mean_std_dict[i]['mean'] = m
    data_types_mean_std_dict[i]['std'] = s
del data_temp_list
print(data_types_mean_std_dict)

with open(save_dir + 'mean_std_dict.pickle', 'wb') as f:
    pickle.dump(data_types_mean_std_dict, f, protocol=4)


##==================== GET NORMALIZE IMAGES
X_train_input = []
X_train_target = []
# X_train_target_whole = [] # 1 2 4
# X_train_target_core = [] # 1 4
# X_train_target_enhance = [] # 4

X_dev_input = []
X_dev_target = []
# X_dev_target_whole = [] # 1 2 4
# X_dev_target_core = [] # 1 4
# X_dev_target_enhance = [] # 4

print(" HGG Validation")
for i in survival_id_dev_HGG:
    all_3d_data = []
    for j in data_types:
        img_path = os.path.join(HGG_data_path, i, i + '_' + j + '.nii.gz')
        img = nib.load(img_path).get_data()
        img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img = img.astype(np.float32)
        all_3d_data.append(img)

    seg_path = os.path.join(HGG_data_path, i, i + '_seg.nii.gz')
    seg_img = nib.load(seg_path).get_data()
    seg_img = np.transpose(seg_img, (1, 0, 2))
    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]), axis=2)
        combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
        combined_array.astype(np.float32)
        X_dev_input.append(combined_array)

        seg_2d = seg_img[:, :, j]
        # whole = np.zeros_like(seg_2d)
        # core = np.zeros_like(seg_2d)
        # enhance = np.zeros_like(seg_2d)
        # for index, x in np.ndenumerate(seg_2d):
        #     if x == 1:
        #         whole[index] = 1
        #         core[index] = 1
        #     if x == 2:
        #         whole[index] = 1
        #     if x == 4:
        #         whole[index] = 1
        #         core[index] = 1
        #         enhance[index] = 1
        # X_dev_target_whole.append(whole)
        # X_dev_target_core.append(core)
        # X_dev_target_enhance.append(enhance)
        seg_2d.astype(int)
        X_dev_target.append(seg_2d)
    del all_3d_data
    gc.collect()
    print("finished {}".format(i))

print(" LGG Validation")
for i in survival_id_dev_LGG:
    all_3d_data = []
    for j in data_types:
        img_path = os.path.join(LGG_data_path, i, i + '_' + j + '.nii.gz')
        img = nib.load(img_path).get_data()
        img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img = img.astype(np.float32)
        all_3d_data.append(img)

    seg_path = os.path.join(LGG_data_path, i, i + '_seg.nii.gz')
    seg_img = nib.load(seg_path).get_data()
    seg_img = np.transpose(seg_img, (1, 0, 2))
    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]), axis=2)
        combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
        combined_array.astype(np.float32)
        X_dev_input.append(combined_array)

        seg_2d = seg_img[:, :, j]
        # whole = np.zeros_like(seg_2d)
        # core = np.zeros_like(seg_2d)
        # enhance = np.zeros_like(seg_2d)
        # for index, x in np.ndenumerate(seg_2d):
        #     if x == 1:
        #         whole[index] = 1
        #         core[index] = 1
        #     if x == 2:
        #         whole[index] = 1
        #     if x == 4:
        #         whole[index] = 1
        #         core[index] = 1
        #         enhance[index] = 1
        # X_dev_target_whole.append(whole)
        # X_dev_target_core.append(core)
        # X_dev_target_enhance.append(enhance)
        seg_2d.astype(int)
        X_dev_target.append(seg_2d)
    del all_3d_data
    gc.collect()
    print("finished {}".format(i))

X_dev_input = np.asarray(X_dev_input, dtype=np.float32)
X_dev_target = np.asarray(X_dev_target)#, dtype=np.float32)
# print(X_dev_input.shape)
# print(X_dev_target.shape)

# with open(save_dir + 'dev_input.pickle', 'wb') as f:
#     pickle.dump(X_dev_input, f, protocol=4)
# with open(save_dir + 'dev_target.pickle', 'wb') as f:
#     pickle.dump(X_dev_target, f, protocol=4)

# del X_dev_input, X_dev_target

print(" HGG Train")
for i in survival_id_tr_HGG:
    all_3d_data = []
    for j in data_types:
        img_path = os.path.join(HGG_data_path, i, i + '_' + j + '.nii.gz')
        img = nib.load(img_path).get_data()
        img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img = img.astype(np.float32)
        all_3d_data.append(img)

    seg_path = os.path.join(HGG_data_path, i, i + '_seg.nii.gz')
    seg_img = nib.load(seg_path).get_data()
    seg_img = np.transpose(seg_img, (1, 0, 2))
    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]), axis=2)
        combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
        combined_array.astype(np.float32)
        X_train_input.append(combined_array)

        seg_2d = seg_img[:, :, j]
        # whole = np.zeros_like(seg_2d)
        # core = np.zeros_like(seg_2d)
        # enhance = np.zeros_like(seg_2d)
        # for index, x in np.ndenumerate(seg_2d):
        #     if x == 1:
        #         whole[index] = 1
        #         core[index] = 1
        #     if x == 2:
        #         whole[index] = 1
        #     if x == 4:
        #         whole[index] = 1
        #         core[index] = 1
        #         enhance[index] = 1
        # X_train_target_whole.append(whole)
        # X_train_target_core.append(core)
        # X_train_target_enhance.append(enhance)
        seg_2d.astype(int)
        X_train_target.append(seg_2d)
    del all_3d_data
    print("finished {}".format(i))
    # print(len(X_train_target))


print(" LGG Train")
for i in survival_id_tr_LGG:
    all_3d_data = []
    for j in data_types:
        img_path = os.path.join(LGG_data_path, i, i + '_' + j + '.nii.gz')
        img = nib.load(img_path).get_data()
        img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img = img.astype(np.float32)
        all_3d_data.append(img)

    seg_path = os.path.join(LGG_data_path, i, i + '_seg.nii.gz')
    seg_img = nib.load(seg_path).get_data()
    seg_img = np.transpose(seg_img, (1, 0, 2))
    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]), axis=2)
        combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
        combined_array.astype(np.float32)
        X_train_input.append(combined_array)

        seg_2d = seg_img[:, :, j]
        # whole = np.zeros_like(seg_2d)
        # core = np.zeros_like(seg_2d)
        # enhance = np.zeros_like(seg_2d)
        # for index, x in np.ndenumerate(seg_2d):
        #     if x == 1:
        #         whole[index] = 1
        #         core[index] = 1
        #     if x == 2:
        #         whole[index] = 1
        #     if x == 4:
        #         whole[index] = 1
        #         core[index] = 1
        #         enhance[index] = 1
        # X_train_target_whole.append(whole)
        # X_train_target_core.append(core)
        # X_train_target_enhance.append(enhance)
        seg_2d.astype(int)
        X_train_target.append(seg_2d)
    del all_3d_data
    print("finished {}".format(i))

X_train_input = np.asarray(X_train_input, dtype=np.float32)
X_train_target = np.asarray(X_train_target)#, dtype=np.float32)
# print(X_train_input.shape)
# print(X_train_target.shape)

# with open(save_dir + 'train_input.pickle', 'wb') as f:
#     pickle.dump(X_train_input, f, protocol=4)
# with open(save_dir + 'train_target.pickle', 'wb') as f:
#     pickle.dump(X_train_target, f, protocol=4)



# X_train_target_whole = np.asarray(X_train_target_whole)
# X_train_target_core = np.asarray(X_train_target_core)
# X_train_target_enhance = np.asarray(X_train_target_enhance)


# X_dev_target_whole = np.asarray(X_dev_target_whole)
# X_dev_target_core = np.asarray(X_dev_target_core)
# X_dev_target_enhance = np.asarray(X_dev_target_enhance)


# print(X_train_target_whole.shape)
# print(X_train_target_core.shape)
# print(X_train_target_enhance.shape)

# print(X_dev_target_whole.shape)
# print(X_dev_target_core.shape)
# print(X_dev_target_enhance.shape)



# with open(save_dir + 'train_target_whole.pickle', 'wb') as f:
#     pickle.dump(X_train_target_whole, f, protocol=4)

# with open(save_dir + 'train_target_core.pickle', 'wb') as f:
#     pickle.dump(X_train_target_core, f, protocol=4)

# with open(save_dir + 'train_target_enhance.pickle', 'wb') as f:
#     pickle.dump(X_train_target_enhance, f, protocol=4)

# with open(save_dir + 'dev_target_whole.pickle', 'wb') as f:
#     pickle.dump(X_dev_target_whole, f, protocol=4)

# with open(save_dir + 'dev_target_core.pickle', 'wb') as f:
#     pickle.dump(X_dev_target_core, f, protocol=4)

# with open(save_dir + 'dev_target_enhance.pickle', 'wb') as f:
#     pickle.dump(X_dev_target_enhance, f, protocol=4)
```

# model.py

```python
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np


from tensorlayer.layers import *
def u_net(x, is_train=False, reuse=False, n_out=1):
    _, nx, ny, nz = x.get_shape().as_list()
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name='inputs')
        conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, name='conv1_1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='conv1_2')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, name='conv2_1')
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='conv2_2')
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, name='conv3_1')
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='conv3_2')
        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, name='conv4_1')
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='conv4_2')
        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, name='conv5_1')
        conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, name='conv5_2')

        up4 = DeConv2d(conv5, 512, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4')
        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, name='uconv4_1')
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='uconv4_2')
        up3 = DeConv2d(conv4, 256, (3, 3), (nx/4, ny/4), (2, 2), name='deconv3')
        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, name='uconv3_1')
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='uconv3_2')
        up2 = DeConv2d(conv3, 128, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')
        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu,  name='uconv2_1')
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='uconv2_2')
        up1 = DeConv2d(conv2, 64, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1')
        up1 = ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, name='uconv1_1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='uconv1_2')
        conv1 = Conv2d(conv1, n_out, (1, 1), act=tf.nn.sigmoid, name='uconv1')
    return conv1

# def u_net(x, is_train=False, reuse=False, pad='SAME', n_out=2):
#     """ Original U-Net for cell segmentataion
#     http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
#     Original x is [batch_size, 572, 572, ?], pad is VALID
#     """
#     from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2d, ConcatLayer
#     nx = int(x._shape[1])
#     ny = int(x._shape[2])
#     nz = int(x._shape[3])
#     print(" * Input: size of image: %d %d %d" % (nx, ny, nz))
#
#     w_init = tf.truncated_normal_initializer(stddev=0.01)
#     b_init = tf.constant_initializer(value=0.0)
#     with tf.variable_scope("u_net", reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#         inputs = InputLayer(x, name='inputs')
#
#         conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv1_1')
#         conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv1_2')
#         pool1 = MaxPool2d(conv1, (2, 2), padding=pad, name='pool1')
#
#         conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv2_1')
#         conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv2_2')
#         pool2 = MaxPool2d(conv2, (2, 2), padding=pad, name='pool2')
#
#         conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv3_1')
#         conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv3_2')
#         pool3 = MaxPool2d(conv3, (2, 2), padding=pad, name='pool3')
#
#         conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv4_1')
#         conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv4_2')
#         pool4 = MaxPool2d(conv4, (2, 2), padding=pad, name='pool4')
#
#         conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv5_1')
#         conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='conv5_2')
#
#         print(" * After conv: %s" % conv5.outputs)
#
#         up4 = DeConv2d(conv5, 512, (3, 3), out_size = (nx/8, ny/8),
#                     strides=(2, 2), padding=pad, act=None,
#                     W_init=w_init, b_init=b_init, name='deconv4')
#         up4 = ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
#         conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv4_1')
#         conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv4_2')
#
#         up3 = DeConv2d(conv4, 256, (3, 3), out_size = (nx/4, ny/4),
#                     strides=(2, 2), padding=pad, act=None,
#                     W_init=w_init, b_init=b_init, name='deconv3')
#         up3 = ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
#         conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv3_1')
#         conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv3_2')
#
#         up2 = DeConv2d(conv3, 128, (3, 3), out_size=(nx/2, ny/2),
#                     strides=(2, 2), padding=pad, act=None,
#                     W_init=w_init, b_init=b_init, name='deconv2')
#         up2 = ConcatLayer([up2, conv2] ,concat_dim=3, name='concat2')
#         conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv2_1')
#         conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv2_2')
#
#         up1 = DeConv2d(conv2, 64, (3, 3), out_size=(nx/1, ny/1),
#                     strides=(2, 2), padding=pad, act=None,
#                     W_init=w_init, b_init=b_init, name='deconv1')
#         up1 = ConcatLayer([up1, conv1] ,concat_dim=3, name='concat1')
#         conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv1_1')
#         conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding=pad,
#                     W_init=w_init, b_init=b_init, name='uconv1_2')
#
#         conv1 = Conv2d(conv1, n_out, (1, 1), act=tf.nn.sigmoid, name='uconv1')
#         print(" * Output: %s" % conv1.outputs)
#
#         # logits0 = conv1.outputs[:,:,:,0]            # segmentataion
#         # logits1 = conv1.outputs[:,:,:,1]            # edge
#         # logits0 = tf.expand_dims(logits0, axis=3)
#         # logits1 = tf.expand_dims(logits1, axis=3)
#     return conv1


def u_net_bn(x, is_train=False, reuse=False, batch_size=None, pad='SAME', n_out=1):
    """image to image translation via conditional adversarial learning"""
    nx = int(x._shape[1])
    ny = int(x._shape[2])
    nz = int(x._shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))

    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name='inputs')

        conv1 = Conv2d(inputs, 64, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv1')
        conv2 = Conv2d(conv1, 128, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv2')
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(conv2, 256, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv3')
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn3')

        conv4 = Conv2d(conv3, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv4')
        conv4 = BatchNormLayer(conv4, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn4')

        conv5 = Conv2d(conv4, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv5')
        conv5 = BatchNormLayer(conv5, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn5')

        conv6 = Conv2d(conv5, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv6')
        conv6 = BatchNormLayer(conv6, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn6')

        conv7 = Conv2d(conv6, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv7')
        conv7 = BatchNormLayer(conv7, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn7')

        conv8 = Conv2d(conv7, 512, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2), padding=pad, W_init=w_init, b_init=b_init, name='conv8')
        print(" * After conv: %s" % conv8.outputs)
        # exit()
        # print(nx/8)
        up7 = DeConv2d(conv8, 512, (4, 4), out_size=(2, 2), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv7')
        up7 = BatchNormLayer(up7, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn7')

        # print(up6.outputs)
        up6 = ConcatLayer([up7, conv7], concat_dim=3, name='concat6')
        up6 = DeConv2d(up6, 1024, (4, 4), out_size=(4, 4), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv6')
        up6 = BatchNormLayer(up6, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn6')
        # print(up6.outputs)
        # exit()

        up5 = ConcatLayer([up6, conv6], concat_dim=3, name='concat5')
        up5 = DeConv2d(up5, 1024, (4, 4), out_size=(8, 8), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv5')
        up5 = BatchNormLayer(up5, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn5')
        # print(up5.outputs)
        # exit()

        up4 = ConcatLayer([up5, conv5] ,concat_dim=3, name='concat4')
        up4 = DeConv2d(up4, 1024, (4, 4), out_size=(15, 15), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = BatchNormLayer(up4, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn4')

        up3 = ConcatLayer([up4, conv4] ,concat_dim=3, name='concat3')
        up3 = DeConv2d(up3, 256, (4, 4), out_size=(30, 30), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = BatchNormLayer(up3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn3')

        up2 = ConcatLayer([up3, conv3] ,concat_dim=3, name='concat2')
        up2 = DeConv2d(up2, 128, (4, 4), out_size=(60, 60), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = BatchNormLayer(up2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn2')

        up1 = ConcatLayer([up2, conv2] ,concat_dim=3, name='concat1')
        up1 = DeConv2d(up1, 64, (4, 4), out_size=(120, 120), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = BatchNormLayer(up1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn1')

        up0 = ConcatLayer([up1, conv1] ,concat_dim=3, name='concat0')
        up0 = DeConv2d(up0, 64, (4, 4), out_size=(240, 240), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv0')
        up0 = BatchNormLayer(up0, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn0')
        # print(up0.outputs)
        # exit()

        out = Conv2d(up0, n_out, (1, 1), act=tf.nn.sigmoid, name='out')

        print(" * Output: %s" % out.outputs)
        # exit()

    return out

## old implementation
# def u_net_2d_64_1024_deconv(x, n_out=2):
#     from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2d, ConcatLayer
#     nx = int(x._shape[1])
#     ny = int(x._shape[2])
#     nz = int(x._shape[3])
#     print(" * Input: size of image: %d %d %d" % (nx, ny, nz))
#
#     w_init = tf.truncated_normal_initializer(stddev=0.01)
#     b_init = tf.constant_initializer(value=0.0)
#     inputs = InputLayer(x, name='inputs')
#
#     conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
#     conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
#     pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')
#
#     conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
#     conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
#     pool2 = MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')
#
#     conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_1')
#     conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_2')
#     pool3 = MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')
#
#     conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_1')
#     conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_2')
#     pool4 = MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')
#
#     conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_1')
#     conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_2')
#
#     print(" * After conv: %s" % conv5.outputs)
#
#     up4 = DeConv2d(conv5, 512, (3, 3), out_size = (nx/8, ny/8), strides = (2, 2),
#                                 padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv4')
#     up4 = ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
#     conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv4_1')
#     conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv4_2')
#
#     up3 = DeConv2d(conv4, 256, (3, 3), out_size = (nx/4, ny/4), strides = (2, 2),
#                                 padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv3')
#     up3 = ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
#     conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv3_1')
#     conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv3_2')
#
#     up2 = DeConv2d(conv3, 128, (3, 3), out_size = (nx/2, ny/2), strides = (2, 2),
#                                 padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv2')
#     up2 = ConcatLayer([up2, conv2] ,concat_dim=3, name='concat2')
#     conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv2_1')
#     conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv2_2')
#
#     up1 = DeConv2d(conv2, 64, (3, 3), out_size = (nx/1, ny/1), strides = (2, 2),
#                                 padding = 'SAME', act=None, W_init=w_init, b_init=b_init, name='deconv1')
#     up1 = ConcatLayer([up1, conv1] ,concat_dim=3, name='concat1')
#     conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1_1')
#     conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1_2')
#
#     conv1 = Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')
#     print(" * Output: %s" % conv1.outputs)
#     outputs = tl.act.pixel_wise_softmax(conv1.outputs)
#     return conv1, outputs
#
#
# def u_net_2d_32_1024_upsam(x, n_out=2):
#     """
#     https://github.com/jocicmarko/ultrasound-nerve-segmentation
#     """
#     from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2d, ConcatLayer
#     batch_size = int(x._shape[0])
#     nx = int(x._shape[1])
#     ny = int(x._shape[2])
#     nz = int(x._shape[3])
#     print(" * Input: size of image: %d %d %d" % (nx, ny, nz))
#     ## define initializer
#     w_init = tf.truncated_normal_initializer(stddev=0.01)
#     b_init = tf.constant_initializer(value=0.0)
#     inputs = InputLayer(x, name='inputs')
#
#     conv1 = Conv2d(inputs, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
#     conv1 = Conv2d(conv1, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
#     pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')
#
#     conv2 = Conv2d(pool1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
#     conv2 = Conv2d(conv2, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
#     pool2 = MaxPool2d(conv2, (2,2), padding='SAME', name='pool2')
#
#     conv3 = Conv2d(pool2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_1')
#     conv3 = Conv2d(conv3, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_2')
#     pool3 = MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')
#
#     conv4 = Conv2d(pool3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_1')
#     conv4 = Conv2d(conv4, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_2')
#     pool4 = MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')
#
#     conv5 = Conv2d(pool4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_1')
#     conv5 = Conv2d(conv5, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_2')
#     pool5 = MaxPool2d(conv5, (2, 2), padding='SAME', name='pool6')
#
#     # hao add
#     conv6 = Conv2d(pool5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_1')
#     conv6 = Conv2d(conv6, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_2')
#
#     print(" * After conv: %s" % conv6.outputs)
#
#     # hao add
#     up7 = UpSampling2dLayer(conv6, (15, 15), is_scale=False, method=1, name='up7')
#     up7 =  ConcatLayer([up7, conv5], concat_dim=3, name='concat7')
#     conv7 = Conv2d(up7, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv7_1')
#     conv7 = Conv2d(conv7, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv7_2')
#
#     # print(nx/8,ny/8) # 30 30
#     up8 = UpSampling2dLayer(conv7, (2, 2), method=1, name='up8')
#     up8 = ConcatLayer([up8, conv4], concat_dim=3, name='concat8')
#     conv8 = Conv2d(up8, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv8_1')
#     conv8 = Conv2d(conv8, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv8_2')
#
#     up9 = UpSampling2dLayer(conv8, (2, 2), method=1, name='up9')
#     up9 = ConcatLayer([up9, conv3] ,concat_dim=3, name='concat9')
#     conv9 = Conv2d(up9, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv9_1')
#     conv9 = Conv2d(conv9, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv9_2')
#
#     up10 = UpSampling2dLayer(conv9, (2, 2), method=1, name='up10')
#     up10 = ConcatLayer([up10, conv2] ,concat_dim=3, name='concat10')
#     conv10 = Conv2d(up10, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv10_1')
#     conv10 = Conv2d(conv10, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv10_2')
#
#     up11 = UpSampling2dLayer(conv10, (2, 2), method=1, name='up11')
#     up11 = ConcatLayer([up11, conv1] ,concat_dim=3, name='concat11')
#     conv11 = Conv2d(up11, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv11_1')
#     conv11 = Conv2d(conv11, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv11_2')
#
#     conv12 = Conv2d(conv11, n_out, (1, 1), act=None, name='conv12')
#     print(" * Output: %s" % conv12.outputs)
#     outputs = tl.act.pixel_wise_softmax(conv12.outputs)
#     return conv10, outputs
#
#
# def u_net_2d_32_512_upsam(x, n_out=2):
#     """
#     https://github.com/jocicmarko/ultrasound-nerve-segmentation
#     """
#     from tensorlayer.layers import InputLayer, Conv2d, MaxPool2d, DeConv2d, ConcatLayer
#     batch_size = int(x._shape[0])
#     nx = int(x._shape[1])
#     ny = int(x._shape[2])
#     nz = int(x._shape[3])
#     print(" * Input: size of image: %d %d %d" % (nx, ny, nz))
#     ## define initializer
#     w_init = tf.truncated_normal_initializer(stddev=0.01)
#     b_init = tf.constant_initializer(value=0.0)
#     inputs = InputLayer(x, name='inputs')
#     # inputs = Input((1, img_rows, img_cols))
#     conv1 = Conv2d(inputs, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
#     # print(conv1.outputs) # (10, 240, 240, 32)
#     # conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
#     conv1 = Conv2d(conv1, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
#     # print(conv1.outputs)    # (10, 240, 240, 32)
#     # conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
#     pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')
#     # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     # print(pool1.outputs)    # (10, 120, 120, 32)
#     # exit()
#     conv2 = Conv2d(pool1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
#     # conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
#     conv2 = Conv2d(conv2, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
#     # conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
#     pool2 = MaxPool2d(conv2, (2,2), padding='SAME', name='pool2')
#     # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
#     conv3 = Conv2d(pool2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_1')
#     # conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
#     conv3 = Conv2d(conv3, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_2')
#     # conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
#     pool3 = MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')
#     # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     # print(pool3.outputs)   # (10, 30, 30, 64)
#
#     conv4 = Conv2d(pool3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_1')
#     # print(conv4.outputs)    # (10, 30, 30, 256)
#     # conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
#     conv4 = Conv2d(conv4, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_2')
#     # print(conv4.outputs)    # (10, 30, 30, 256) != (10, 30, 30, 512)
#     # conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
#     pool4 = MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')
#     # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#     conv5 = Conv2d(pool4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_1')
#     # conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
#     conv5 = Conv2d(conv5, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_2')
#     # conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
#     # print(conv5.outputs)    # (10, 15, 15, 512)
#     print(" * After conv: %s" % conv5.outputs)
#     # print(nx/8,ny/8) # 30 30
#     up6 = UpSampling2dLayer(conv5, (2, 2), name='up6')
#     # print(up6.outputs)  # (10, 30, 30, 512) == (10, 30, 30, 512)
#     up6 = ConcatLayer([up6, conv4], concat_dim=3, name='concat6')
#     # print(up6.outputs)  # (10, 30, 30, 768)
#     # up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
#     conv6 = Conv2d(up6, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_1')
#     # conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
#     conv6 = Conv2d(conv6, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv6_2')
#     # conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)
#
#     up7 = UpSampling2dLayer(conv6, (2, 2), name='up7')
#     up7 = ConcatLayer([up7, conv3] ,concat_dim=3, name='concat7')
#     # up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
#     conv7 = Conv2d(up7, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv7_1')
#     # conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
#     conv7 = Conv2d(conv7, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv7_2')
#     # conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)
#
#     up8 = UpSampling2dLayer(conv7, (2, 2), name='up8')
#     up8 = ConcatLayer([up8, conv2] ,concat_dim=3, name='concat8')
#     # up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
#     conv8 = Conv2d(up8, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv8_1')
#     # conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
#     conv8 = Conv2d(conv8, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv8_2')
#     # conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
#
#     up9 = UpSampling2dLayer(conv8, (2, 2), name='up9')
#     up9 = ConcatLayer([up9, conv1] ,concat_dim=3, name='concat9')
#     # up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
#     conv9 = Conv2d(up9, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv9_1')
#     # conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
#     conv9 = Conv2d(conv9, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv9_2')
#     # conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
#
#     conv10 = Conv2d(conv9, n_out, (1, 1), act=None, name='conv9')
#     # conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
#     print(" * Output: %s" % conv10.outputs)
#     outputs = tl.act.pixel_wise_softmax(conv10.outputs)
#     return conv10, outputs


if __name__ == "__main__":
    pass
    # main()
```

# train.py

```python
#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os, time, model

def distort_imgs(data):
    """ data augumentation """
    x1, x2, x3, x4, y = data
    # x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],  # previous without this, hard-dice=83.7
    #                         axis=0, is_random=True) # up down
    x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],
                            axis=1, is_random=True) # left right
    x1, x2, x3, x4, y = tl.prepro.elastic_transform_multi([x1, x2, x3, x4, y],
                            alpha=720, sigma=24, is_random=True)
    x1, x2, x3, x4, y = tl.prepro.rotation_multi([x1, x2, x3, x4, y], rg=20,
                            is_random=True, fill_mode='constant') # nearest, constant
    x1, x2, x3, x4, y = tl.prepro.shift_multi([x1, x2, x3, x4, y], wrg=0.10,
                            hrg=0.10, is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.shear_multi([x1, x2, x3, x4, y], 0.05,
                            is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.zoom_multi([x1, x2, x3, x4, y],
                            zoom_range=[0.9, 1.1], is_random=True,
                            fill_mode='constant')
    return x1, x2, x3, x4, y

def vis_imgs(X, y, path):
    """ show one slice """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
        X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
        X[:,:,3,np.newaxis], y]), size=(1, 5),
        image_path=path)

def vis_imgs2(X, y_, y, path):
    """ show one slice with target """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    if y_.ndim == 2:
        y_ = y_[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
        X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
        X[:,:,3,np.newaxis], y_, y]), size=(1, 6),
        image_path=path)

def main(task='all'):
    ## Create folder to save trained model and result images
    save_dir = "checkpoint"
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir("samples/{}".format(task))

    ###======================== LOAD DATA ===================================###
    ## by importing this, you can load a training set and a validation set.
    # you will get X_train_input, X_train_target, X_dev_input and X_dev_target
    # there are 4 labels in targets:
    # Label 0: background
    # Label 1: necrotic and non-enhancing tumor
    # Label 2: edema
    # Label 4: enhancing tumor
    import prepare_data_with_valid as dataset
    X_train = dataset.X_train_input
    y_train = dataset.X_train_target[:,:,:,np.newaxis]
    X_test = dataset.X_dev_input
    y_test = dataset.X_dev_target[:,:,:,np.newaxis]

    if task == 'all':
        y_train = (y_train > 0).astype(int)
        y_test = (y_test > 0).astype(int)
    elif task == 'necrotic':
        y_train = (y_train == 1).astype(int)
        y_test = (y_test == 1).astype(int)
    elif task == 'edema':
        y_train = (y_train == 2).astype(int)
        y_test = (y_test == 2).astype(int)
    elif task == 'enhance':
        y_train = (y_train == 4).astype(int)
        y_test = (y_test == 4).astype(int)
    else:
        exit("Unknow task %s" % task)

    ###======================== HYPER-PARAMETERS ============================###
    batch_size = 10
    lr = 0.0001 
    # lr_decay = 0.5
    # decay_every = 100
    beta1 = 0.9
    n_epoch = 100
    print_freq_step = 100

    ###======================== SHOW DATA ===================================###
    # show one slice
    X = np.asarray(X_train[80])
    y = np.asarray(y_train[80])
    # print(X.shape, X.min(), X.max()) # (240, 240, 4) -0.380588 2.62761
    # print(y.shape, y.min(), y.max()) # (240, 240, 1) 0 1
    nw, nh, nz = X.shape
    vis_imgs(X, y, 'samples/{}/_train_im.png'.format(task))
    # show data augumentation results
    for i in range(10):
        x_flair, x_t1, x_t1ce, x_t2, label = distort_imgs([X[:,:,0,np.newaxis], X[:,:,1,np.newaxis],
                X[:,:,2,np.newaxis], X[:,:,3,np.newaxis], y])#[:,:,np.newaxis]])
        # print(x_flair.shape, x_t1.shape, x_t1ce.shape, x_t2.shape, label.shape) # (240, 240, 1) (240, 240, 1) (240, 240, 1) (240, 240, 1) (240, 240, 1)
        X_dis = np.concatenate((x_flair, x_t1, x_t1ce, x_t2), axis=2)
        # print(X_dis.shape, X_dis.min(), X_dis.max()) # (240, 240, 4) -0.380588233471 2.62376139209
        vis_imgs(X_dis, label, 'samples/{}/_train_im_aug{}.png'.format(task, i))

    with tf.device('/cpu:0'):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.device('/gpu:0'): #<- remove it if you train on CPU or other GPU
            ###======================== DEFIINE MODEL =======================###
            ## nz is 4 as we input all Flair, T1, T1c and T2.
            t_image = tf.placeholder('float32', [batch_size, nw, nh, nz], name='input_image')
            ## labels are either 0 or 1
            t_seg = tf.placeholder('float32', [batch_size, nw, nh, 1], name='target_segment')
            ## train inference
            net = model.u_net(t_image, is_train=True, reuse=False, n_out=1)
            ## test inference
            net_test = model.u_net(t_image, is_train=False, reuse=True, n_out=1)

            ###======================== DEFINE LOSS =========================###
            ## train losses
            out_seg = net.outputs
            dice_loss = 1 - tl.cost.dice_coe(out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
            iou_loss = tl.cost.iou_coe(out_seg, t_seg, axis=[0,1,2,3])
            dice_hard = tl.cost.dice_hard_coe(out_seg, t_seg, axis=[0,1,2,3])
            loss = dice_loss

            ## test losses
            test_out_seg = net_test.outputs
            test_dice_loss = 1 - tl.cost.dice_coe(test_out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
            test_iou_loss = tl.cost.iou_coe(test_out_seg, t_seg, axis=[0,1,2,3])
            test_dice_hard = tl.cost.dice_hard_coe(test_out_seg, t_seg, axis=[0,1,2,3])

        ###======================== DEFINE TRAIN OPTS =======================###
        t_vars = tl.layers.get_variables_with_name('u_net', True, True)
        with tf.device('/gpu:0'):
            with tf.variable_scope('learning_rate'):
                lr_v = tf.Variable(lr, trainable=False)
            train_op = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=t_vars)

        ###======================== LOAD MODEL ==============================###
        tl.layers.initialize_global_variables(sess)
        ## load existing model if possible
        tl.files.load_and_assign_npz(sess=sess, name=save_dir+'/u_net_{}.npz'.format(task), network=net)

        ###======================== TRAINING ================================###
    for epoch in range(0, n_epoch+1):
        epoch_time = time.time()
        ## update decay learning rate at the beginning of a epoch
        # if epoch !=0 and (epoch % decay_every == 0):
        #     new_lr_decay = lr_decay ** (epoch // decay_every)
        #     sess.run(tf.assign(lr_v, lr * new_lr_decay))
        #     log = " ** new learning rate: %f" % (lr * new_lr_decay)
        #     print(log)
        # elif epoch == 0:
        #     sess.run(tf.assign(lr_v, lr))
        #     log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
        #     print(log)

        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        for batch in tl.iterate.minibatches(inputs=X_train, targets=y_train,
                                    batch_size=batch_size, shuffle=True):
            images, labels = batch
            step_time = time.time()
            ## data augumentation for a batch of Flair, T1, T1c, T2 images
            # and label maps synchronously.
            data = tl.prepro.threading_data([_ for _ in zip(images[:,:,:,0, np.newaxis],
                    images[:,:,:,1, np.newaxis], images[:,:,:,2, np.newaxis],
                    images[:,:,:,3, np.newaxis], labels)],
                    fn=distort_imgs) # (10, 5, 240, 240, 1)
            b_images = data[:,0:4,:,:,:]  # (10, 4, 240, 240, 1)
            b_labels = data[:,4,:,:,:]
            b_images = b_images.transpose((0,2,3,1,4))
            b_images.shape = (batch_size, nw, nh, nz)

            ## update network
            _, _dice, _iou, _diceh, out = sess.run([train_op,
                    dice_loss, iou_loss, dice_hard, net.outputs],
                    {t_image: b_images, t_seg: b_labels})
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1

            ## you can show the predition here:
            # vis_imgs2(b_images[0], b_labels[0], out[0], "samples/{}/_tmp.png".format(task))
            # exit()

            # if _dice == 1: # DEBUG
            #     print("DEBUG")
            #     vis_imgs2(b_images[0], b_labels[0], out[0], "samples/{}/_debug.png".format(task))

            if n_batch % print_freq_step == 0:
                print("Epoch %d step %d 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)"
                % (epoch, n_batch, _dice, _diceh, _iou, time.time()-step_time))

            ## check model fail
            if np.isnan(_dice):
                exit(" ** NaN loss found during training, stop training")
            if np.isnan(out).any():
                exit(" ** NaN found in output images during training, stop training")

        print(" ** Epoch [%d/%d] train 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)" %
                (epoch, n_epoch, total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch, time.time()-epoch_time))

        ## save a predition of training set
        for i in range(batch_size):
            if np.max(b_images[i]) > 0:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/train_{}.png".format(task, epoch))
                break
            elif i == batch_size-1:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/train_{}.png".format(task, epoch))

        ###======================== EVALUATION ==========================###
        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        for batch in tl.iterate.minibatches(inputs=X_test, targets=y_test,
                                        batch_size=batch_size, shuffle=True):
            b_images, b_labels = batch
            _dice, _iou, _diceh, out = sess.run([test_dice_loss,
                    test_iou_loss, test_dice_hard, net_test.outputs],
                    {t_image: b_images, t_seg: b_labels})
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1

        print(" **"+" "*17+"test 1-dice: %f hard-dice: %f iou: %f (2d no distortion)" %
                (total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch))
        print(" task: {}".format(task))
        ## save a predition of test set
        for i in range(batch_size):
            if np.max(b_images[i]) > 0:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/test_{}.png".format(task, epoch))
                break
            elif i == batch_size-1:
                vis_imgs2(b_images[i], b_labels[i], out[i], "samples/{}/test_{}.png".format(task, epoch))

        ###======================== SAVE MODEL ==========================###
        tl.files.save_npz(net.all_params, name=save_dir+'/u_net_{}.npz'.format(task), sess=sess)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='all', help='all, necrotic, edema, enhance')

    args = parser.parse_args()

    main(args.task)
```
