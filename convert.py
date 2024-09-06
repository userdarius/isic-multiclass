import onnx
from onnx_tf.backend import prepare

# Load your ONNX model
onnx_model = onnx.load('super_resolution.onnx')

# Convert
tf_rep = prepare(onnx_model)

# Export the model to a TensorFlow format
tf_rep.export_graph('converted_model.pb')
