# -*- coding: utf-8 -*-
from . import *
from . import loss
from . import metric
from . import model

import tensorflow as tf

model_dict = {'RecGOBD_1_embedding':model.RecGOBD_1_embedding,'RecGOBD_2_embeddings':model.RecGOBD_2_embeddings,
             'RecGOBD_3_embeddings':model.RecGOBD_3_embeddings,'RecGOBD_4_embeddings':model.RecGOBD_4_embeddings}
loss_dict = {'NLL': tf.keras.losses.BinaryCrossentropy, 'Focal': loss.BinaryFocalloss}
