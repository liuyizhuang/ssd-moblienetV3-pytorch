from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
#from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite
import sys
import torch.onnx
from onnx_tf.backend import prepare
#from caffe2.python.onnx.backend import Caffe2Backend as c2
import onnx
import time
from PIL import Image
import torchvision
import tensorflow as tf
if len(sys.argv) < 3:
    print('Usage: python convert_to_caffe2_models.py <net type: mobilenet-v1-ssd|others>  <model path>')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]

label_path = sys.argv[3]

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb3-ssd-lite':
    net = create_mobilenetv3_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)
device = torch.device("cpu")
net.eval()
net =net.to(device)
#print(net,model_path)

model_path1 = "mb3-ssd.onnx"
init_net_path = f"models/{net_type}_init_net.pb"
init_net_txt_path = f"models/{net_type}_init_net.pbtxt"
predict_net_path = f"models/{net_type}_predict_net.pb"
predict_net_txt_path = f"models/{net_type}_predict_net.pbtxt"

dummy_input = torch.randn(1, 3, 300, 300)
dummy_input = dummy_input.to(device)
#time.sleep( 5 )
#print ("start : %s" % time.ctime())
#time.sleep(1)
torch.onnx.export(net, dummy_input, model_path1, output_names=['scores', 'boxes'])

print("111111111111111111111111111111111111111111111111111111111")
#print ("End : %s" % time.ctime())
#print(net_type,model_path)
onnx_model = onnx.load(model_path1)


#onnx_model = onnx.load(onnx_filename)
tf_rep = prepare(onnx_model, strict=False)
# install onnx-tensorflow from github，and tf_rep = prepare(onnx_model, strict=False)
# Reference https://github.com/onnx/onnx-tensorflow/issues/167
#tf_rep = prepare(onnx_model) # whthout strict=False leads to KeyError: 'pyfunc_0'
image = Image.open('pants.jpg')
loader = torchvision.transforms.ToTensor()
image1 = loader(image).unsqueeze(0)
image1 =image1.to(device)  
# debug, here using the same input to check onnx and tf.
output_pytorch, img_np = net(dummy_input)
print('output_pytorch = {}'.format(output_pytorch),type(img_np),img_np)
img_np = img_np.detach().numpy()
output_onnx_tf = tf_rep.run(dummy_input)
print('output_onnx_tf = {}'.format(output_onnx_tf))
# onnx --> tf.graph.pb
tf_pb_path = 'ft_graph.pb'
tf_rep.export_graph(tf_pb_path)

with tf.Graph().as_default():
        graph_def = tf.GraphDef()
        with open(tf_pb_path, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        #for op in graph_def.get_operations():
        #    print(op.name)
        with tf.Session() as sess:
            #init = tf.initialize_all_variables()
            init = tf.global_variables_initializer()
            #sess.run(init)
            
            # print all ops, check input/output tensor name.
            # uncomment it if you donnot know io tensor names.
            '''
            print('-------------ops---------------------')
            op = sess.graph.get_operations()
            for m in op:
                print(m.values())
            print('-------------ops done.---------------------')
            '''

            input_x = sess.graph.get_tensor_by_name("0:0") # input
            outputs1 = sess.graph.get_tensor_by_name('Add_1:0') # 5
            outputs2 = sess.graph.get_tensor_by_name('add_3:0') # 10
            box,score,class1 = sess.run([outputs1, outputs2], feed_dict={input_x:dummy_input})
            #output_tf_pb = sess.run([outputs1, outputs2], feed_dict={input_x:np.random.randn(1, 3, 224, 224)})
            print('output_tf_pb = {}'.format(box),sorce,"_______________________________".class1)
'''
init_net, predict_net = c2.onnx_graph_to_caffe2_net(model)

print(f"Save the model in binary format to the files {init_net_path} and {predict_net_path}.")

with open(init_net_path, "wb") as fopen:
    fopen.write(init_net.SerializeToString())
with open(predict_net_path, "wb") as fopen:
    fopen.write(predict_net.SerializeToString())

print(f"Save the model in txt format to the files {init_net_txt_path} and {predict_net_txt_path}. ")
with open(init_net_txt_path, 'w') as f:
    f.write(str(init_net))

with open(predict_net_txt_path, 'w') as f:
    f.write(str(predict_net))
'''
