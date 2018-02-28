import matplotlib.pyplot as plt
from matplotlib.image import imread
import tensorflow as tf
import numpy as np
import sys
import os

# Load Data
import knifey

from knifey import img_size, img_size_flat, img_shape, num_classes, num_channels

# 在您的计算机上设置用于存储数据集的目录。
# knifey.data_dir = "data/knifey-spoony/"
# Knifey-Spoony数据集大约为22 MB，如果它不在给定路径中，它将自动下载。
knifey.maybe_download_and_extract()

dataset = knifey.load()
class_names = dataset.class_names
# class_names ['forky', 'knifey', 'spoony']

# Training and Test-Sets
image_paths_train, cls_train, labels_train = dataset.get_training_set()

# Print the first image-path to see if it looks OK.
# image_paths_train[0]
# '/home/magnus/development/TensorFlow-Tutorials/data/knifey-spoony/forky/forky-05-0023.jpg'

# Get the test-set.
image_paths_test, cls_test, labels_test = dataset.get_test_set()

# 函数用于在3x3网格中绘制9个图像，并在每个图像下面写入真实和预测的类。
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name,
                                                       cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# 该数据集不加载实际图像，而是具有训练集中图像的列表和测试集中图像的另一个列表。 这个辅助函数加载一些图像文件。

def load_images(image_paths):
    # Load the images from disk.
    images = [imread(path) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)

# Plot a few images to see if data is correct

# Load the first images from the test-set.
images = load_images(image_paths=image_paths_test[0:9])

# Get the true classes for those images.
cls_true = cls_test[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true, smooth=True)

# Create TFRecords
path_tfrecords_train = os.path.join(knifey.data_dir, "train.tfrecords")
# path_tfrecords_train # 'data/knifey-spoony/train.tfrecords'

path_tfrecords_test = os.path.join(knifey.data_dir, "test.tfrecords")
# path_tfrecords_test

# 帮助功能打印转换进度。
def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

# 用于包装整数的辅助函数，以便将其保存到TFRecords文件中。
def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Helper函数用于封装原始字节，以便将它们保存到TFRecords文件中。
def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

'''
这是从磁盘读取图像并将它们与类标签一起写入TFRecords文件的功能。 
这会将图像加载并解码为numpy数组，然后将原始字节存储在TFRecords文件中。 
如果原始图像文件被压缩，例如 作为jpeg文件，那么TFRecords文件可能比原始图像文件大很多倍。

也可以将压缩的图像文件直接保存在TFRecords文件中，因为它可以保存任何原始字节。 
当稍后在下面的parse（）函数中读取TFRecords文件时，我们将不得不解码压缩的图像。
'''


def convert(image_paths, labels, out_path):
    # Args:
    # image_paths   List of file-paths for the images.
    # labels        Class-labels for the images.
    # out_path      File-path for the TFRecords output file.

    print("Converting: " + out_path)

    # Number of images. Used when printing the progress.
    num_images = len(image_paths)

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
        # Iterate over all the image-paths and class-labels.
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            # Print the percentage-progress.
            print_progress(count=i, total=num_images - 1)

            # Load the image-file using matplotlib's imread function.
            img = imread(path)

            # Convert the image to raw bytes.
            img_bytes = img.tostring()

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                {
                    'image': wrap_bytes(img_bytes),
                    'label': wrap_int64(label)
                }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)

'''
请注意将数据字典写入TFRecords文件所需的4个函数调用。 
在Google Developers的原始代码示例中，这4个函数调用实际上是嵌套的。 
TensorFlow的设计哲学一般似乎是这样的：如果一个函数调用是好的，那么4个函数调用是4倍，如果它们嵌套，那么它是指数级的善良！
当然，这是非常糟糕的API设计，因为最后一个函数writer.write（）应该能够直接获取数据dict，然后在内部调用其他3个函数。
将训练集转换为TFRecords文件。 请注意，我们如何使用整数类数字作为标签而不是单热编码数组。
'''
convert(image_paths=image_paths_train,
        labels=cls_train,
        out_path=path_tfrecords_train)

# 将测试集转换为TFRecords文件：
convert(image_paths=image_paths_test,
        labels=cls_test,
        out_path=path_tfrecords_test)

# 估计器的输入函数
'''
TFRecords文件包含序列化二进制格式的数据，
需要将其转换回正确数据类型的图像和标签。 我们在这个解析中使用了一个辅助函数：
'''


def parse(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = \
        {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    image_raw = parsed_example['image']

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.decode_raw(image_raw, tf.uint8)

    # The type is now uint8 but we need it to be float.
    image = tf.cast(image, tf.float32)

    # Get the label associated with the image.
    label = parsed_example['label']

    # The image and label are now correct TensorFlow types.
    return image, label

# 帮助函数用于创建从TFRecords文件中读取以用于Estimator API的输入函数。
def input_fn(filenames, train, batch_size=32, buffer_size=2048):
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # train:       Boolean whether training (True) or testing (False).
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=filenames)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(parse)

    if train:
        # If training then read a buffer of the given size and
        # randomly shuffle it.
        dataset = dataset.shuffle(buffer_size=buffer_size)

        # Allow infinite reading of the data.
        num_repeat = None
    else:
        # If testing then don't shuffle the data.

        # Only go through the data once.
        num_repeat = 1

    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    images_batch, labels_batch = iterator.get_next()

    # The input-function must return a dict wrapping the images.
    x = {'image': images_batch}
    y = labels_batch

    return x, y

# 这是用于Estimator API的训练集的输入函数：
def train_input_fn():
    return input_fn(filenames=path_tfrecords_train, train=True)

# 这是用于Estimator API的测试集的输入函数：
def test_input_fn():
    return input_fn(filenames=path_tfrecords_test, train=False)

# 用于预测新图像的输入函数
'''
预测新数据的类别也需要输入函数。 作为一个例子，我们只是使用测试集中的一些图像。
你可以在这里加载你想要的任何图像。 确保它们与TensorFlow模型的尺寸相同，否则您需要调整图像大小。
'''
some_images = load_images(image_paths=image_paths_test[0:9])
'''
这些图像现在作为numpy数组存储在内存中，因此我们可以使用Estimator API的标准输入函数。 
请注意，图像以uint8数据加载，但它必须以浮点形式输入到TensorFlow图形中，因此我们进行了类型转换。
'''
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image": some_images.astype(np.float32)},
    num_epochs=1,
    shuffle=False)

some_images_cls = cls_test[0:9]

# 使用预先制作的估算器时，我们需要指定数据的输入要素。 在这种情况下，我们希望从我们的数据集中输入图像，这些图像是给定形状的数值数组。
feature_image = tf.feature_column.numeric_column("image",
                                                 shape=img_shape)
# 您可以有几个输入功能，然后将它们合并到一个列表中：
feature_columns = [feature_image]

# 这个例子中，我们想要分别使用512,256和128个单元的3层DNN。、
num_hidden_units = [512, 256, 128]

# DNNClassifier然后为我们构建神经网络。 我们也可以指定激活函数和其他各种参数（请参阅文档）。
# 这里我们只是指定类的数量和检查点将被保存的目录。
model = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                   hidden_units=num_hidden_units,
                                   activation_fn=tf.nn.relu,
                                   n_classes=num_classes,
                                   model_dir="./checkpoints_tutorial18-1/")

# Training
model.train(input_fn=train_input_fn, steps=200)

# Evaluation
result = model.evaluate(input_fn=test_input_fn)
print(result)

print("Classification accuracy: {0:.2%}".format(result["accuracy"]))

# Predictions
predictions = model.predict(input_fn=predict_input_fn)
cls = [p['classes'] for p in predictions]
cls_pred = np.array(cls, dtype='int').squeeze()
print(cls_pred)


plot_images(images=some_images,
            cls_true=some_images_cls,
            cls_pred=cls_pred)

# 整个测试集的预测
predictions = model.predict(input_fn=test_input_fn)
cls = [p['classes'] for p in predictions]
cls_pred = np.array(cls, dtype='int').squeeze()

print(np.sum(cls_pred == 2))








