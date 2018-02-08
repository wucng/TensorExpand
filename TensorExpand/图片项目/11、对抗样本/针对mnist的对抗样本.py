# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

train=False # 训练还是推理
Confused=True # 是否混淆 最后的分类结果 如：加入噪声让7 识别成1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

with tf.variable_scope('D'):
    # Set model weights
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Construct model
    y_pred = tf.nn.softmax(tf.matmul(x, W)+b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pred), reduction_indices=1))

train_op=tf.train.AdamOptimizer().minimize(cost)

init = tf.global_variables_initializer()

saver=tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    sess.run(init)

    if train:
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Fit training using batch data
                # _, _, c = sess.run([new_W, new_b, cost], feed_dict={x: batch_xs, y: batch_ys})

                _,c=sess.run([train_op,cost],feed_dict={x: batch_xs, y: batch_ys})

                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        saver.save(sess,'./models/model.ckpt')

        # test
        acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1)),tf.float32))
        print('test acc',acc.eval({x: mnist.test.images, y: mnist.test.labels}))

        print("Optimization Finished!")
    else:
        saver.restore(sess,'./models/model.ckpt')


        input_image=mnist.test.images[:1]
        # print(input_image.shape)
        real_label=mnist.test.labels[:1]

        if not Confused:
            pred_label=y_pred.eval({x:input_image})

            print('pred',np.argmax(pred_label,1),'\n','real',np.argmax(real_label,1))
            # pred [7]
            # real [7]
        else:

            with tf.get_default_graph().as_default():
                # Add a placeholder variable for the target class-number.
                # This will be set to e.g. 300 for the 'bookcase' class.
                pl_cls_target = tf.placeholder(dtype=tf.int32)

                # Add a new loss-function. This is the cross-entropy.
                # See Tutorial #01 for an explanation of cross-entropy.
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=[pl_cls_target])

                # Get the gradient for the loss-function with regard to
                # the resized input image.
                gradient = tf.gradients(loss, x)


            def find_adversary_noise(image_path,cls_target, noise_limit=3.0,
                                     required_score=0.99, max_iterations=100):
                """
                Find the noise that must be added to the given image so
                that it is classified as the target-class.

                image_path: File-path to the input-image (must be *.jpg).
                cls_target: Target class-number (integer between 1-1000).
                noise_limit: Limit for pixel-values in the noise.
                required_score: Stop when target-class score reaches this.
                max_iterations: Max number of optimization iterations to perform.
                """

                # Create a feed-dict with the image.
                feed_dict = {x:image_path}

                # Use TensorFlow to calculate the predicted class-scores
                # (aka. probabilities) as well as the resized image.
                pred, image = sess.run([y_pred, x],
                                          feed_dict=feed_dict)

                # Convert to one-dimensional array.
                pred = np.squeeze(pred)

                # Predicted class-number.
                cls_source = np.argmax(pred)

                # Score for the predicted class (aka. probability or confidence).
                score_source_org = pred.max()

                name_source='7'
                name_target='1'

                # Initialize the noise to zero.
                noise = 0

                # Perform a number of optimization iterations to find
                # the noise that causes mis-classification of the input image.
                for i in range(max_iterations):
                    print("Iteration:", i)

                    # The noisy image is just the sum of the input image and noise.
                    noisy_image = image + noise

                    # Ensure the pixel-values of the noisy image are between
                    # 0 and 255 like a real image. If we allowed pixel-values
                    # outside this range then maybe the mis-classification would
                    # be due to this 'illegal' input breaking the Inception model.
                    noisy_image = np.clip(a=noisy_image, a_min=0.0, a_max=255.0)

                    # Create a feed-dict. This feeds the noisy image to the
                    # tensor in the graph that holds the resized image, because
                    # this is the final stage for inputting raw image data.
                    # This also feeds the target class-number that we desire.
                    feed_dict = {x: noisy_image,
                                 pl_cls_target: cls_target}

                    # Calculate the predicted class-scores as well as the gradient.
                    pred, grad = sess.run([y_pred, gradient],
                                             feed_dict=feed_dict)

                    # Convert the predicted class-scores to a one-dim array.
                    pred = np.squeeze(pred)

                    # The scores (probabilities) for the source and target classes.
                    score_source = pred[cls_source]
                    score_target = pred[cls_target]

                    # Squeeze the dimensionality for the gradient-array.
                    grad = np.array(grad).squeeze()

                    # The gradient now tells us how much we need to change the
                    # noisy input image in order to move the predicted class
                    # closer to the desired target-class.

                    # Calculate the max of the absolute gradient values.
                    # This is used to calculate the step-size.
                    grad_absmax = np.abs(grad).max()

                    # If the gradient is very small then use a lower limit,
                    # because we will use it as a divisor.
                    if grad_absmax < 1e-10:
                        grad_absmax = 1e-10

                    # Calculate the step-size for updating the image-noise.
                    # This ensures that at least one pixel colour is changed by 7.
                    # Recall that pixel colours can have 255 different values.
                    # This step-size was found to give fast convergence.
                    step_size = 7 / grad_absmax

                    # Print the score etc. for the source-class.
                    msg = "Source score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
                    print(msg.format(score_source, cls_source, name_source))

                    # Print the score etc. for the target-class.
                    msg = "Target score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
                    print(msg.format(score_target, cls_target, name_target))

                    # Print statistics for the gradient.
                    msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
                    print(msg.format(grad.min(), grad.max(), step_size))

                    # Newline.
                    print()

                    # If the score for the target-class is not high enough.
                    if score_target < required_score:
                        # Update the image-noise by subtracting the gradient
                        # scaled by the step-size.
                        noise -= step_size * grad

                        # Ensure the noise is within the desired range.
                        # This avoids distorting the image too much.
                        noise = np.clip(a=noise,
                                        a_min=-noise_limit,
                                        a_max=noise_limit)
                    else:
                        # Abort the optimization because the score is high enough.
                        break

                return image.squeeze(), noisy_image.squeeze(), noise, \
                       name_source, name_target, \
                       score_source, score_source_org, score_target


            # 绘制图像和噪声的帮助函数
            # 函数对输入做归一化，则输入值在0.0到1.0之间，这样才能正确的显示出噪声。
            def normalize_image(x):
                # Get the min and max values for all pixels in the input.
                x_min = x.min()
                x_max = x.max()

                # Normalize so all values are between 0.0 and 1.0
                x_norm = (x - x_min) / (x_max - x_min)

                return x_norm


            # 这个函数绘制了原始图像、噪声图像，以及噪声。它也显示了类别名和评分。
            def plot_images(image, noise, noisy_image,
                            name_source, name_target,
                            score_source, score_source_org, score_target):
                """
                Plot the image, the noisy image and the noise.
                Also shows the class-names and scores.

                Note that the noise is amplified to use the full range of
                colours, otherwise if the noise is very low it would be
                hard to see.

                image: Original input image.
                noise: Noise that has been added to the image.
                noisy_image: Input image + noise.
                name_source: Name of the source-class.
                name_target: Name of the target-class.
                score_source: Score for the source-class.
                score_source_org: Original score for the source-class.
                score_target: Score for the target-class.
                """

                # Create figure with sub-plots.
                fig, axes = plt.subplots(1, 3, figsize=(10, 10))

                # Adjust vertical spacing.
                fig.subplots_adjust(hspace=0.1, wspace=0.1)

                # Use interpolation to smooth pixels?
                smooth = True

                # Interpolation type.
                if smooth:
                    interpolation = 'spline16'
                else:
                    interpolation = 'nearest'

                # Plot the original image.
                # Note that the pixel-values are normalized to the [0.0, 1.0]
                # range by dividing with 255.
                ax = axes.flat[0]
                # ax.imshow(image / 255.0, interpolation=interpolation)
                ax.imshow(image.reshape([28,28]) , interpolation=interpolation)
                msg = "Original Image:\n{0} ({1:.2%})"
                xlabel = msg.format(name_source, score_source_org)
                ax.set_xlabel(xlabel)

                # Plot the noisy image.
                ax = axes.flat[1]
                # ax.imshow(noisy_image / 255.0, interpolation=interpolation)
                ax.imshow(noisy_image.reshape([28,28]), interpolation=interpolation)
                msg = "Image + Noise:\n{0} ({1:.2%})\n{2} ({3:.2%})"
                xlabel = msg.format(name_source, score_source, name_target, score_target)
                ax.set_xlabel(xlabel)

                # Plot the noise.
                # The colours are amplified otherwise they would be hard to see.
                ax = axes.flat[2]
                ax.imshow(normalize_image(noise.reshape([28,28])), interpolation=interpolation)
                xlabel = "Amplified Noise"
                ax.set_xlabel(xlabel)

                # Remove ticks from all the plots.
                for ax in axes.flat:
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Ensure the plot is shown correctly with multiple plots
                # in a single Notebook cell.
                plt.show()


            # 寻找并绘制对抗样本的帮助函数
            # 这个函数结合了上面的两个方法。它先找到对抗噪声，然后画出图像和噪声。
            def adversary_example(image_path, cls_target,
                                  noise_limit, required_score):
                """
                Find and plot adversarial noise for the given image.

                image_path: File-path to the input-image (must be *.jpg).
                cls_target: Target class-number (integer between 1-1000).
                noise_limit: Limit for pixel-values in the noise.
                required_score: Stop when target-class score reaches this.
                """

                # Find the adversarial noise.
                image, noisy_image, noise, \
                name_source, name_target, \
                score_source, score_source_org, score_target = \
                    find_adversary_noise(image_path=image_path,
                                         cls_target=cls_target,
                                         noise_limit=noise_limit,
                                         required_score=required_score)

                # Plot the image and the noise.
                plot_images(image=image, noise=noise, noisy_image=noisy_image,
                            name_source=name_source, name_target=name_target,
                            score_source=score_source,
                            score_source_org=score_source_org,
                            score_target=score_target)

                # Print some statistics for the noise.
                msg = "Noise min: {0:.3f}, max: {1:.3f}, mean: {2:.3f}, std: {3:.3f}"
                print(msg.format(noise.min(), noise.max(),
                                 noise.mean(), noise.std()))


            '''
            噪声界限设为3.0，这表示只允许每个像素颜色在3.0范围内波动。
            像素颜色在0到255之间，因此3.0的浮动对应大约1.2%的可能范围。
            这样的少量噪声对人眼是不可见的，因此噪声图像和原始图像看起来基本一致，如下所示。
            '''

            '''
            要求评分设为0.99，这表示当目标分类的评分大于等于0.99时，用来寻找对抗噪声的优化器就会停止，
            这样Inception模型几乎确定了噪声图像展示的是期望的目标类别。
            '''
            image_path = "images/parrot_cropped1.jpg"

            adversary_example(image_path=input_image,
                              cls_target=1,
                              noise_limit=3.0,
                              required_score=0.99)
