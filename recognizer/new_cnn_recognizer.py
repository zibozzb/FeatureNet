#########################################################################
'''
    Author: Zhibo Zhang
    MADlab, Mechanical Enineering, University at Buffalo
    Time: 2017.8.30
    Github: https://github.com/zibozzb
    MADlab: http://madlab.eng.buffalo.edu
    
    '''
#########################################################################

import os
import os.path
import math
import numpy as np
import tensorflow as tf
import new_input_processor


import datetime
starttime = datetime.datetime.now()

BATCH_SIZE = 40
learning_rate = 0.001
MAX_STEP = 20000

def inference(models):
 
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',shape=[7, 7, 7, 1, 32],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases',shape=[32],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv3d(models, weights, strides=[1, 2, 2, 2, 1], padding='SAME')
        batch_norm = tf.contrib.layers.batch_norm(conv,data_format='NHWC', center=True,scale=True,scope='batch_norm')
        pre_activation = tf.nn.bias_add(batch_norm, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations", conv1)

    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',shape=[5, 5, 5, 32, 32],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[32],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv3d(conv1, weights, strides=[1, 1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations", conv2)

    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable('weights',shape=[4, 4, 4, 32, 64],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[64],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv3d(conv2, weights, strides=[1, 1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name='conv3')
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations", conv3)

    # with tf.variable_scope('pooling1') as scope:
    #     pool1 = tf.nn.max_pool3d(conv3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],padding='SAME', name='pooling2')

    with tf.variable_scope('conv4') as scope:
        weights = tf.get_variable('weights',shape=[3, 3, 3, 64, 64],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[64],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv3d(conv3, weights, strides=[1, 1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name='conv4')
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations", conv4)

    # pool2
    with tf.variable_scope('pooling2') as scope:
        # norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool3d(conv4, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],padding='SAME', name='pooling2')

    #local
    with tf.variable_scope('local') as scope:
        reshape = tf.reshape(pool2, shape=[BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',shape=[dim, 128],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases',shape=[128],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        local = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',shape=[128, 24],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases',shape=[24],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local, weights), biases, name='softmax_linear')

    return softmax_linear


# %%

def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        labels = tf.cast(labels, tf.int64)

        # to use this loss fuction, one-hot encoding is needed!
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example')

        #        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
        #                        (logits=logits, labels=labels, name='xentropy_per_example')

        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)

    return loss

def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        labels = tf.argmax(labels, 1)  # one_hot decode
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy

"""
def train():
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    # starter_learning_rate = 0.001
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, my_global_step,
    #                                            4000, 0.2, staircase=True)# changing LR

    data_dir = 'D:\\1_big_data_selected\\binvox\\all\\rot\\whole\\train\\dataset\\'
    log_dir = 'D:\\1_big_data_selected\\binvox\\all\\rot\\10_class\\train\\dataset\\log1\\'

    images, labels = new_input_processor.read_cifar10(data_dir=data_dir,
                                                is_train=True,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)
    logits = inference(images)

    loss = losses(logits, labels)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=my_global_step)
    train__acc = evaluation(logits, labels)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, loss_value,tra_acc = sess.run([train_op, loss, train__acc])
            # _, loss_value = sess.run([train_op, loss])


            if step % 50 == 0:
                print('Step: %d, loss: %.4f,train accuracy = %.2f%%' %(step, loss_value,tra_acc * 100.0))
                # print('Step: %d, loss: %.4f' %(step, loss_value))


            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

train()
"""

'''
###########################train with validation#################################
def train_val():
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    # starter_learning_rate = 0.001
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, my_global_step,
    #                                            4000, 0.2, staircase=True)# changing LR

    data_dir = 'D:\\1_big_data_selected\\binvox\\all\\rot\\whole\\train\\dataset\\'
    val_dir='D:\\1_big_data_selected\\binvox\\all\\rot\\whole\\validation\\dataset\\'
    log_train_dir = 'D:\\1_big_data_selected\\binvox\\all\\rot\\whole\\train\\dataset\\log2\\log_train\\'
    log_val_dir = 'D:\\1_big_data_selected\\binvox\\all\\rot\\whole\\train\\dataset\\log2\\log_val\\'

    images, labels = new_input_processor.read_cifar10(data_dir=data_dir,
                                                      is_train=True,
                                                    batch_size=BATCH_SIZE,
                                                       shuffle=True)
    images_val,labels_val=new_input_processor.read_cifar10(data_dir=val_dir,
                                                           is_train=False,
                                                         batch_size=BATCH_SIZE,
                                                            shuffle=False)


    logits = inference(images)
    loss = losses(logits, labels)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=my_global_step)
    accuracy = evaluation(logits, labels)

    x=tf.placeholder(tf.float32,shape=[BATCH_SIZE,64,64,64,1])
    y_=tf.placeholder(tf.int32, shape=[BATCH_SIZE,24])

    # embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
    # assignment = embedding.assign(embedding_input)


    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_writer = tf.summary.FileWriter(log_train_dir, sess.graph)
    val_writer=tf.summary.FileWriter(log_val_dir,sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            tra_image,tra_label=sess.run([images,labels])
            _, tra_loss,tra_acc = sess.run([train_op, loss, accuracy],
                                             feed_dict={x:tra_image,y_:tra_label})
            # _, loss_value = sess.run([train_op, loss])


            if step % 50 == 0:
                print('Step: %d, loss: %.4f,train accuracy = %.2f%%' %(step, tra_loss,tra_acc * 100.0))
                # print('Step: %d, loss: %.4f' %(step, loss_value))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)


            if step % 200 == 0:
                val_image,val_label = sess.run([images_val,labels_val])
                val_loss,val_acc = sess.run([loss,accuracy],
                                            feed_dict={x:val_image,y_:val_label})
                print('<  Step: %d, valid loss: %.4f,valid accuracy = %.2f%%  >' % (step, val_loss, val_acc * 100.0))
                summary_str = sess.run(summary_op)
                val_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(log_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


train_val()
'''
##################################################################################


'''
def evaluate():
    with tf.Graph().as_default():

        log_dir = 'D:\\1_big_data_selected\\binvox\\all\\rot\\whole\\train\\dataset\\log2\\log_train\\'
        test_dir = 'D:\\1_big_data_selected\\binvox\\all\\rot\\whole\\test\\dataset\\'
        n_test = 21600

        # reading test data
        images, labels = new_input_processor.read_cifar10(data_dir=test_dir,
                                                    is_train=False,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False)

        logits = inference(images)
        labels = tf.argmax(labels, 1) # one_hot decode
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                num_iter = int(math.ceil(n_test / BATCH_SIZE))
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0

                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1
                    precision = true_count / total_sample_count
                print('precision = %.3f' % precision)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

                # %%

evaluate()
'''

'''
def evaluate_ONE():
    with tf.Graph().as_default():

        log_dir = 'D:\\1_big_data_selected\\binvox\\all\\train\\dataset\\log2\\log_train\\'
        test_dir = 'D:\\1_big_data_selected\\stl\\test_multi_feature\\rotation\\test\\bin_file\\1\\test\\'
        # n_test = 3200

        # reading test data
        images, labels = new_input_processor.read_cifar10(data_dir=test_dir,
                                                    is_train=False,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False)

        logits = inference(images)
        # labels = tf.argmax(labels, 1) # one_hot decode
        # top_k_op = tf.nn.in_top_k(logits, labels, 1)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                # num_iter = int(math.ceil(n_test / BATCH_SIZE))
                # true_count = 0
                # total_sample_count = num_iter * BATCH_SIZE
                step = 0

                while step < 1 and not coord.should_stop():
                    predictions,true_label = sess.run([logits,labels])
                    step += 1
                    # precision = true_count / total_sample_count
                    # 'Step: %d, loss: %.4f,train accuracy = %.2f%%' % (step, loss_value, tra_acc * 100.0)
                print('prediction:')
                print(predictions[0:23])
                print('true_label:')
                print(true_label[0])
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
    return predictions[0:24]
                # %%
a=evaluate_ONE()
print(a==np.max(a))
rot,label=np.where(a==np.max(a))
print(np.shape(a))
print(np.where(a==np.max(a)))
'''




############ evaluate 24 rotation###################


'''
def evaluate_rotation(test_dir):
    with tf.Graph().as_default():

        log_dir = 'D:\\1_big_data_selected\\binvox\\all\\train\\dataset\\log2\\log_train\\'
        test_dir = 'D:\\1_big_data_selected\\binvox\\test_one\\rotation\\6\\'
        # n_test = 3200

        # reading test data
        images, labels = new_input_processor.read_cifar10(data_dir=test_dir,
                                                    is_train=False,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False)

        logits = inference(images)
        # labels = tf.argmax(labels, 1) # one_hot decode
        # top_k_op = tf.nn.in_top_k(logits, labels, 1)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                # num_iter = int(math.ceil(n_test / BATCH_SIZE))
                # true_count = 0
                # total_sample_count = num_iter * BATCH_SIZE
                step = 0

                while step < 1 and not coord.should_stop():
                    predictions,true_label = sess.run([logits,labels])
                    step += 1
                    # precision = true_count / total_sample_count
                    # 'Step: %d, loss: %.4f,train accuracy = %.2f%%' % (step, loss_value, tra_acc * 100.0)
                # print('prediction:')
                # print(predictions[0])
                # print('true_label:')
                # print(true_label[0])
                return predictions[0]
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

prediction_same_part=np.zeros((24,24))
prediction_all=np.array([])
for part_num in range(5,6):
    for rot_num in range(1,25):
        input_path="'D:\\1_big_data_selected\\stl\\test_multi_feature\\rotation\\test\\bin_file\\1\\test\\'"+str(part_num)+"\\"+str(rot_num)+"\\"
        prediction_same_part[rot_num-1,:]=evaluate_rotation(input_path)
        # print(evaluate_rotation(input_path))
        # print(np.where(evaluate_rotation(input_path)==np.max(evaluate_rotation(input_path))))
    print(prediction_same_part)
    rot,label=np.where(prediction_same_part==np.max(prediction_same_part))
    np.append(prediction_all,label)
print(label)

# '''

#################################### Confusion Matrix ########################################################
import scipy.io as scio
def confusion_matrix():
    with tf.Graph().as_default():

        log_dir = 'D:\\1_big_data_selected\\binvox\\all\\rot\\whole\\train\\dataset\\log2\\log_train\\'
        test_dir = 'D:\\1_big_data_selected\\binvox\\all\\rot\\whole\\test\\dataset\\'
        n_test = 21600

        # reading test data
        images, labels = new_input_processor.read_cifar10(data_dir=test_dir,
                                                    is_train=False,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False)

        logits = inference(images)
        labels = tf.argmax(labels, 1) # one_hot decode
        value,id=tf.nn.top_k(logits)
        predict=id
        # top_k_op = tf.nn.in_top_k(logits, labels, 1)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                num_iter = int(math.ceil(n_test / BATCH_SIZE))
                print(num_iter)
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0
                predict_label_all=np.zeros((40,1))
                true_label_all=np.zeros((1,40))
                while step < num_iter and not coord.should_stop():
                    predict_label,true_label=sess.run([predict,labels])
                    predict_label_all=np.hstack((predict_label_all,predict_label))
                    true_label_all=np.vstack((true_label_all,true_label))
                    # print(predict_label)
                    print(true_label)
                    # predictions = sess.run([top_k_op])
                    # true_count += np.sum(predictions)
                    step += 1
                    # precision = true_count / total_sample_count
                # print('precision = %.3f' % precision)
                return np.reshape(np.transpose(predict_label_all[:,1:]),(1,21600)),np.reshape(true_label_all[1:,:],(1,21600))
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

                # %%

predict_all,true_all=confusion_matrix()
save_dir = 'D:\\1_big_data_selected\\share\\con_matrix'
scio.savemat(save_dir, {'predict':predict_all,'true':true_all})



# print(np.shape(predict_all))
# print(np.shape(true_all))
