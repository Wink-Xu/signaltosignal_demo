思路 and 笔记

1.文件夹设置。
2.子程序
  （1）数据预处理
  （2）建立前向传播 inference: 
             tf.get_variables(name,shape,dtype,initializer)  tf.Variable(constant,name) 变量
             tf.placeholder(dtype,shape,name)
  （3）loss\optimizer\global_variable_initializer\sess.run\learning_rate\batch_size

	batch 这里有一个除不尽的问题 有些数据没用到
  （4）summary（tensorboard） / ckpt(save_model)

      双斜杠，即tensorboard --logdir=E://MyTensorBoard//logs

  （5）一边train 一边evaluate 
3.数据

  ①. 40000张图片 通过关联成像仿真 得到 40000个 1024维信号  √
  ②. 每个信号单独归一化，到0~255.  将所有信号相加 再进行排序 得到256张图片的序号 √
  ③. 通过序号图片进行成像  (检查） √ （这种排序效果一般，因为数据太多太不同 40000张图综合排序）
  ④. 给相应的序号的 [1,256]维 信号增加噪声 （加性噪声 乘性噪声）
  ⑤. 通过深度学习给信号去噪 。 （这里有个疑问，我们已知hadamard图片，信号到图片只需要进行关联计算即可）
   