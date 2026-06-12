import os
if 'TF_USE_LEGACY_KERAS' in os.environ:
    del os.environ['TF_USE_LEGACY_KERAS']

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
import numpy as np

original_call = Dense.__call__
def patched_call(self, inputs, *args, **kwargs):
    if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
        inputs = inputs[0]
    return original_call(self, inputs, *args, **kwargs)
Dense.__call__ = patched_call

model = load_model('backend/models/garbage_tf_model.h5', compile=False)

x = np.random.rand(1, 224, 224, 3).astype('float32')

# Manual inference
out = x
for layer in model.layers:
    out = layer(out, training=False)
    if isinstance(out, (list, tuple)):
        out = out[0]

print("Manual inference successful, output shape:", out.shape)
