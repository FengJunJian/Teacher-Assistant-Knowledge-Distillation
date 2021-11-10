from openvino.inference_engine import IECore

import numpy as np
import cv2

import time
from PIL import Image
# from ImageNetClass import class_name
from data_loader import ship_classes
from torchvision import transforms
import onnxruntime



onnx_model = 'resnet110.onnx'#"resnet8.onnx"#
model_xml="openvino/resnet110.xml"
model_bin="openvino/resnet110.bin"

imgPath='../Classification_advanced/00vessel/MVI_1469_VIS_00003_1.jpg'
imgPath='../Classification_advanced/03Buoy/MVI_1469_VIS_00109_3.jpg'
###onnx
session=onnxruntime.InferenceSession(onnx_model)

###openvino
ie = IECore()
net = ie.read_network(model=model_xml,weights=model_bin)
exec_net = ie.load_network(network=net, device_name="CPU")
input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))
net.batch_size = 1  # batchsize

###opencv
# net=cv2.dnn.Net_readFromModelOptimizer(model_xml,model_bin)


mean=(0.51264606, 0.55715489, 0.6386575)
std=(0.15772002, 0.14560729, 0.13691749)
normalize = transforms.Normalize(mean=(0.51264606, 0.55715489, 0.6386575), std=(0.15772002, 0.14560729, 0.13691749))
transform = transforms.Compose([transforms.Resize((32, 64)),transforms.RandomCrop(32, padding=4),
										   transforms.ToTensor(), normalize])

n, c, h, w = net.input_info[input_blob].input_data.shape
print(n, c, h, w)
images = np.ndarray(shape=(n, c, h, w),dtype=np.float32)

#dst=np.empty((h,w,c),dtype=np.float32)
for i in range(n):
    image = cv2.imread(imgPath)#(h,w)
    #im=Image.open(imgPath)
    if image.shape[:-1] != (32, 64):
        image = cv2.resize(image, (64, 32))
    image=image[:,16:48]
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    cv2.normalize(image, image, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    # im=Image.open(imgPath)
    # im=transform(im)
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    images[i] = image

# im=im.numpy()
# im=np.reshape(im,(1,)+im.shape)


input_dict={input_blob:images}
start = time.time()
onnx_output=session.run(None,input_dict)
end=time.time()
print('onnx runtime is %.10f s' % (end - start))
# print(onnx_output[0],ship_classes[onnx_output[0].argmax(1)[0]])
print(ship_classes[onnx_output[0].argmax(1)[0]])

# time.sleep(2)
start = time.time()
res = exec_net.infer(inputs={input_blob: images})
end=time.time()
print('OpenVINO total time is %.10f s' % (end - start))
# print(res[out_blob],ship_classes[res[out_blob].argmax(1)[0]])
print(ship_classes[res[out_blob].argmax(1)[0]])



