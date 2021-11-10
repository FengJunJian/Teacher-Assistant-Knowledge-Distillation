import argparse
import os
from thop import profile
import torch
import torchvision
from torchvision import transforms
import onnx
import onnxruntime
from PIL import Image
from ImageNetClass import class_name
from model_factory import create_cnn_model
from data_loader import get_ship
import time
from openvino.inference_engine import IECore
#####################################################################################
def alexnet_demo():
    img=Image.open('../Classification_advanced/00vessel/MVI_1469_VIS_00003_1.jpg')
    img=transform(img)
    img=torch.unsqueeze(img,0)
    dummy_input = torch.randn(1, 3, 224, 224, )#device="cuda"
    model = build_model2onnx(img)  #build_model2onnx
    o1 = model(dummy_input)
    o2 = model(img)
    input1 = {"actual_input_1": img.numpy()}
    filename = 'alexnet.onnx'
    output = inference(filename, input1) #inference

def build_model2onnx(input1):
    '''
    input1:input for the model
    '''
    model = torchvision.models.alexnet(pretrained=True)#.cuda() resnet18
    input_names = [ "input" ] #+ #[ "learned_%d" % i for i in range(16) ]
    output_names = [ "output" ]
    torch.onnx.export(model, input1, "alexnet.onnx", export_params=True,verbose=True, input_names=input_names, output_names=output_names)
    return model

def inference(filename,input_dict):
    # model_onnx = onnx.load("alexnet.onnx")
    # onnx.checker.check_model(model_onnx)
    # # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model_onnx.graph))
    session=onnxruntime.InferenceSession(filename)
    start=time.time()
    output=session.run(None,input_dict)
    end=time.time()
    return output,end-start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TA Knowledge Distillation Code')
    parser.add_argument('--student', '--model', default='resnet110', type=str, help='teacher student name')#resnet110  resnet8 resnet56
    parser.add_argument('--dataset-dir', default='../Classification_advanced', type=str, help='dataset directory')
    parser.add_argument('--checkpoint', default='experiment/resnet110_110-56.pth', type=str, help='Checkpoint for model')#resnet110_110-56.pth resnet8_56-8.pth resnet56_56-56.pth
    args = parser.parse_args()
    print(args)
    normalize = transforms.Normalize(mean=(0.51264606, 0.55715489, 0.6386575), std=(0.15772002, 0.14560729, 0.13691749))
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),normalize])

    num_classes = 14
    student_model = create_cnn_model(args.student, use_cuda=False,num_classes=num_classes)
    student_model.eval()

    if not os.path.exists(args.checkpoint):
        raise Exception('No checkpoint')
    batch_size=3
    train_loader, test_loader = get_ship(num_classes=num_classes,dataset_dir=args.dataset_dir,batch_size=batch_size)
    it=train_loader.__iter__()
    data,target=it.next()

    # for i, (data,target) in enumerate(train_loader):
    #     print(i,data.shape)

    model_weight=torch.load(args.checkpoint)
    student_model.load_state_dict(model_weight['model_state_dict'])
    # student_model(data)
    input_names = [ "input1" ]
    output_names = [ "output1" ]
    onnx_model_name=args.student+"batch.onnx"
    if False or not os.path.exists(onnx_model_name):
        torch.onnx.export(student_model, data, onnx_model_name, export_params=True,verbose=True, input_names=input_names, output_names=output_names)

    flops, params = profile(student_model, inputs=(data,))
    print('FLOPs = ' + str(flops / 1000 ** 2) + 'M')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    ##########################################inference
    input1 = {"input1": data.numpy()}
    start=time.time()
    output2 = student_model(data)
    end = time.time()
    print('time:%f s'%(end-start))
    # start=time.time()
    output1,time_int=inference(onnx_model_name,input1)
    # end=time.time()
    print('time onnx:%f s'%time_int)

    ###openvino
    #weights = model_bin
    ie = IECore()
    net = ie.read_network(model=onnx_model_name)#onnx_model_name
    #basename=onnx_model_name.split('.')[0]
    #net = ie.read_network(model=os.path.join('openvino',basename+'16.xml'),weights=os.path.join('openvino',basename+'16.bin'))  #
    exec_net = ie.load_network(network=net, device_name="CPU")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    net.batch_size = batch_size  # batchsize
    start = time.time()
    res = exec_net.infer(inputs={input_blob: data.numpy()})
    end = time.time()
    print('OpenVINO total time is %.10f s' % (end - start))
    # print(res[out_blob],ship_classes[res[out_blob].argmax(1)[0]])
    #print(ship_classes[res[out_blob].argmax(1)[0]])




