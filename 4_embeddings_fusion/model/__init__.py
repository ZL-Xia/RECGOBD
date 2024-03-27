# -*- coding: utf-8 -*-
from . import *
from . import loss
from . import metric
from . import model

import tensorflow as tf

model_dict = {'DeepAtt': model.DeepAtt, 'DeepAttPlus': model.DeepAttPlus, 'DeepSEA': model.DeepSEA,
              'DanQ': model.DanQ, 'DanQ_JASPAR': model.DanQ_JASPAR,'DeepAtt_4_embedding': model.DeepAtt_4_embedding}
loss_dict = {'NLL': tf.keras.losses.BinaryCrossentropy, 'Focal': loss.BinaryFocalloss}
