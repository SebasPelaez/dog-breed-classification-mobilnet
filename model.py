import tensorflow as tf

import utils

class DepthwiseSeparableConvolution(tf.keras.models.Model):
  
  def __init__(self,conv_filters,conv_strides,width_multiplier, **kwargs):
    super(DepthwiseSeparableConvolution, self).__init__(**kwargs)
    
    self.dw_filter = conv_filters[0]
    self.pw_filter = conv_filters[1]
    
    self.dw_stride = conv_strides[0]
    self.pw_stride = conv_strides[1]
    
    self.width_multiplier = width_multiplier
    
  def build(self, input_shape):
    
    point_wise_filters = round(self.pw_filter * self.width_multiplier)
    
    self.depthwise_conv = tf.keras.layers.SeparableConv2D(
        filters=self.dw_filter,
        kernel_size=3,
        strides=self.dw_stride,
        padding='same',
        depth_multiplier=self.width_multiplier)
    self.bn_dw = tf.keras.layers.BatchNormalization()
    self.activation_dw = tf.keras.layers.Activation('relu')

    self.pointwise_conv = tf.keras.layers.Conv2D(
        filters=point_wise_filters,
        kernel_size=1,
        strides=self.pw_stride,
        padding='same')
    self.bn_pw = tf.keras.layers.BatchNormalization()
    self.activation_pw = tf.keras.layers.Activation('relu')

  def call(self, inputs, training=None):
    
    dw_conv = self.depthwise_conv(inputs)
    dw_conv = self.bn_dw(dw_conv, training=training)
    dw_conv = self.activation_dw(dw_conv)

    pw_conv = self.pointwise_conv(dw_conv)
    pw_conv = self.bn_pw(pw_conv, training=training)
    pw_conv = self.activation_pw(pw_conv)

    return pw_conv

class MobilNet_Architecture(tf.keras.models.Model):
  
  def __init__(self,width_multiplier, **kwargs):
    super(MobilNet_Architecture, self).__init__(**kwargs)
    self.width_multiplier = width_multiplier
        
  def build(self, input_shape):

    self.conv1 = tf.keras.layers.Conv2D(filters=32,kernel_size=3,strides=2,padding='same')
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.activation1 = tf.keras.layers.Activation('relu')
    
    self.depthwise_block1 = DepthwiseSeparableConvolution(conv_filters=(32,64),conv_strides=(1,1),width_multiplier=self.width_multiplier)
    self.depthwise_block2 = DepthwiseSeparableConvolution(conv_filters=(64,128),conv_strides=(2,1),width_multiplier=self.width_multiplier)
    self.depthwise_block3 = DepthwiseSeparableConvolution(conv_filters=(128,128),conv_strides=(1,1),width_multiplier=self.width_multiplier)
    self.depthwise_block4 = DepthwiseSeparableConvolution(conv_filters=(128,256),conv_strides=(2,1),width_multiplier=self.width_multiplier)
    self.depthwise_block5 = DepthwiseSeparableConvolution(conv_filters=(256,256),conv_strides=(1,1),width_multiplier=self.width_multiplier)
    self.depthwise_block6 = DepthwiseSeparableConvolution(conv_filters=(256,512),conv_strides=(2,1),width_multiplier=self.width_multiplier)
    
    self.dw_penta_block = [
        DepthwiseSeparableConvolution(conv_filters=(512,512),conv_strides=(1,1),width_multiplier=self.width_multiplier)
        for _ in range(5)
    ]       
    
    self.depthwise_block7 = DepthwiseSeparableConvolution(conv_filters=(512,1024),conv_strides=(2,1),width_multiplier=self.width_multiplier)
    self.depthwise_block8 = DepthwiseSeparableConvolution(conv_filters=(1024,1024),conv_strides=(1,1),width_multiplier=self.width_multiplier)
    
    self.global_average_pool = tf.keras.layers.GlobalAveragePooling2D()
    self.fully_connected = tf.keras.layers.Dense(units = 120)
        
  def call(self, inputs, training=None):

    first_conv = self.conv1(inputs)
    first_conv = self.bn1(first_conv, training=training)
    first_conv = self.activation1(first_conv)
    
    dw_con1 = self.depthwise_block1(first_conv, training=training)
    dw_con2 = self.depthwise_block2(dw_con1, training=training)
    dw_con3 = self.depthwise_block3(dw_con2, training=training)
    dw_con4 = self.depthwise_block4(dw_con3, training=training)
    dw_con5 = self.depthwise_block5(dw_con4, training=training)
    dw_con6 = self.depthwise_block6(dw_con5, training=training)
    
    penta_block = dw_con6
    for dw in self.dw_penta_block:
        penta_block = dw(penta_block, training = training)
        
    
    dw_con7 = self.depthwise_block7(penta_block, training=training)
    dw_con8 = self.depthwise_block8(dw_con7, training=training)
    
    gap = self.global_average_pool(dw_con8)
    output = self.fully_connected(gap)

    return output

def preprocess_image(x):
  return tf.cast(x, tf.float32) / 255

def model_fn(features, labels, mode, params):

  training = mode == tf.estimator.ModeKeys.TRAIN

  preprocessed_images = preprocess_image(features['image'])
  batch_label = labels

  model = MobilNet_Architecture(width_multiplier=params['width_multiplier'])
  logits = model(preprocessed_images, training=training)

  y_pred = tf.argmax(input=logits, axis=-1)
  predictions = {
    "classes": y_pred,
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  xentropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  loss = xentropy + tf.losses.get_regularization_loss()

  eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
  }

  tf.summary.scalar('accuracy', tf.metrics.accuracy(labels, y_pred)[1])
  
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      optimizer = tf.train.AdamOptimizer(
        learning_rate=params['learning_rate'],
        beta1=0.9, 
        beta2=0.999, 
        epsilon=1e-08, 
        use_locking=False
      )
      
      train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step()
      )

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)