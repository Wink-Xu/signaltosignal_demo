˼· and �ʼ�

1.�ļ������á�
2.�ӳ���
  ��1������Ԥ����
  ��2������ǰ�򴫲� inference: 
             tf.get_variables(name,shape,dtype,initializer)  tf.Variable(constant,name) ����
             tf.placeholder(dtype,shape,name)
  ��3��loss\optimizer\global_variable_initializer\sess.run\learning_rate\batch_size

	batch ������һ�������������� ��Щ����û�õ�
  ��4��summary��tensorboard�� / ckpt(save_model)

      ˫б�ܣ���tensorboard --logdir=E://MyTensorBoard//logs

  ��5��һ��train һ��evaluate 
3.����

  ��. 40000��ͼƬ ͨ������������� �õ� 40000�� 1024ά�ź�  ��
  ��. ÿ���źŵ�����һ������0~255.  �������ź���� �ٽ������� �õ�256��ͼƬ����� ��
  ��. ͨ�����ͼƬ���г���  (��飩 �� ����������Ч��һ�㣬��Ϊ����̫��̫��ͬ 40000��ͼ�ۺ�����
  ��. ����Ӧ����ŵ� [1,256]ά �ź��������� ���������� ����������
  ��. ͨ�����ѧϰ���ź�ȥ�� �� �������и����ʣ�������֪hadamardͼƬ���źŵ�ͼƬֻ��Ҫ���й������㼴�ɣ�
   