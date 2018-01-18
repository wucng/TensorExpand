# -*- coding: utf-8 -*-

import tensorflow as tf
import midi
import numpy as np
import os

lower_bound = 24
upper_bound = 102
span = upper_bound - lower_bound  # 78


# midi文件转Note(音符)
def midiToNoteStateMatrix(midi_file_path, squash=True, span=span):
    pattern = midi.read_midifile(midi_file_path)  # 读取 .midi文件

    time_left = []
    for track in pattern:
        time_left.append(track[0].tick)

    posns = [0 for track in pattern]  # posns = [0]*len(pattern)

    statematrix = []
    time = 0

    state = [[0, 0] for x in range(span)]  # state =[[0,0]]*span
    statematrix.append(state)
    condition = True
    while condition:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            oldstate = state
            state = [[oldstate[x][0], 0] for x in range(span)]
            statematrix.append(state)
        for i in range(len(time_left)):
            if not condition:
                break
            while time_left[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lower_bound) or (evt.pitch >= upper_bound):
                        pass
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch - lower_bound] = [0, 0]
                        else:
                            state[evt.pitch - lower_bound] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        out = statematrix
                        condition = False
                        break
                try:
                    time_left[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    time_left[i] = None

            if time_left[i] is not None:
                time_left[i] -= 1

        if all(t is None for t in time_left):
            break

        time += 1

    S = np.array(statematrix)
    statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
    statematrix = np.asarray(statematrix).tolist()  # 转成列表
    return statematrix


# Note转midi文件
def noteStateMatrixToMidi(statematrix, filename="output_file", span=span):
    statematrix = np.array(statematrix)
    if not len(statematrix.shape) == 3:
        statematrix = np.dstack((statematrix[:, :span], statematrix[:, span:]))
    statematrix = np.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    span = upper_bound - lower_bound
    tickscale = 55

    lastcmdtime = 0
    prevstate = [[0, 0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time - lastcmdtime) * tickscale, pitch=note + lower_bound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time - lastcmdtime) * tickscale, velocity=40, pitch=note + lower_bound))
            lastcmdtime = time

        prevstate = state

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(filename), pattern)


# 读取midi数据
def get_songs(midi_path):
    files = os.listdir(midi_path)
    songs = []
    for f in files:
        f = midi_path + '/' + f  # f=glob.glob(midi_path+'/*.midi')
        print('Loading:', f)
        try:
            song = np.array(midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 64:
                songs.append(song)
        except Exception as e:
            print('数据无效: ', e)
    print("读取的有效midi文件个数: ", len(songs))
    return songs


# music目录中包含了下载的midi文件
songs = get_songs('./music')

note_range = upper_bound - lower_bound
# 音乐长度
n_timesteps = 128
n_input = 2 * note_range * n_timesteps # 2*(102-24)*128=19968
n_hidden = 64

X = tf.placeholder(tf.float32, [None, n_input]) # [None,19968]
W = None
bh = None
bv = None


def sample(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


def gibbs_sample(k):
    def body(count, k, xk):
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh))
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv))
        return count + 1, k, xk

    count = tf.constant(0)

    def condition(count, k, xk):
        return count < k

    [_, _, x_sample] = tf.while_loop(condition, body, [count, tf.constant(k), X])

    x_sample = tf.stop_gradient(x_sample) # [-1,19968]
    return x_sample


# 定义神经网络
def neural_network():
    global W
    W = tf.Variable(tf.random_normal([n_input, n_hidden], 0.01)) # [19968,64]
    global bh
    bh = tf.Variable(tf.zeros([1, n_hidden], tf.float32)) # [1,64]
    global bv
    bv = tf.Variable(tf.zeros([1, n_input], tf.float32)) # [1,19968]

    x_sample = gibbs_sample(1) # [-1,19968]
    h = sample(tf.sigmoid(tf.matmul(X, W) + bh)) # [-1,64]
    h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh)) # [-1,64]

    learning_rate = tf.constant(0.005, tf.float32)
    size_bt = tf.cast(tf.shape(X)[0], tf.float32) # batch_size
    W_adder = tf.multiply(learning_rate / size_bt,
                        tf.subtract(tf.matmul(tf.transpose(X), h), tf.matmul(tf.transpose(x_sample), h_sample))) # [19968,64]
    bv_adder = tf.multiply(learning_rate / size_bt, tf.reduce_sum(tf.subtract(X, x_sample), 0, True)) # [1,19968]
    bh_adder = tf.multiply(learning_rate / size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True)) # [1,64]
    update = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]
    return update


# 训练神经网络
def train_neural_network():
    update = neural_network()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())

        epochs = 256
        batch_size = 64
        for epoch in range(epochs):
            for song in songs:
                song = np.array(song)
                song = song[:int(np.floor(song.shape[0] / n_timesteps) * n_timesteps)]
                song = np.reshape(song, [song.shape[0] // n_timesteps, song.shape[1] * n_timesteps]) # [8,19968]

                for i in range(1, len(song), batch_size):
                    train_x = song[i:i + batch_size] # [7,19968]
                    sess.run(update, feed_dict={X: train_x})
            print(epoch)
            # 保存模型
            if epoch == epochs - 1:
                saver.save(sess, 'midi.module')

        # 生成midi
        sample = gibbs_sample(1).eval(session=sess, feed_dict={X: np.zeros((1, n_input))})
        S = np.reshape(sample[0, :], (n_timesteps, 2 * note_range))
        noteStateMatrixToMidi(S, "auto_gen_music")
        print('生成auto_gen_music.mid文件')


train_neural_network()
