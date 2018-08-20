from models import *
from grad_cam import *
def flatten_model(model):
	#print(model)
	flattend = []
	for name, module in model._modules.items():
		for name_, module_ in module._modules.items():
				for name__, module__ in module_._modules.items():
					for name___, module___ in module__._modules.items():
						print(name___, module___)
						flattend.append(module___)
	return nn.Sequential(*flattend)
if __name__=='__main__':
	#model   = VGGNet()
	#vggnet   = models.vgg19(pretrained=True)
	densenet   = DenseNet121()
	#print(densenet.features[0])
	#flat_dense = flatten_model(densenet.features)
	#FeatureExtractor(model, ["35"])
	#print(densenet.features[0][10][-1][5])
	#x_test = Variable(torch.randn(1,3,224,224))
	#out = flat_dense(Variable(x_test))
	#print(densenet.features)
	#extractor = ModelOutputs(densenet, ['0'])
	#features, output = extractor(Variable(x_test))
	#print(features, output)
	#index = np.argmax(output.cpu().data.numpy())
	#print(index)
	image_path = 'data/images/CXR2811_IM-1238-2001.png'
	grad_cam = GradCam(model = densenet,target_layer_names = ["0"], use_cuda=False)

	img = cv2.imread(image_path, 1)
	img = np.float32(cv2.resize(img, (224, 224))) / 255
	input = preprocess_image(img)

	# If None, returns the map for the highest scoring category.
	# Otherwise, targets the requested index.
	target_index = None

	mask = grad_cam(input, target_index)

	show_cam_on_image(img, mask)

	gb_model = GuidedBackpropReLUModel(model = densenet, use_cuda=False)
	gb = gb_model(input, index=target_index)
	utils.save_image(torch.from_numpy(gb), 'gb.jpg')

	cam_mask = np.zeros(gb.shape)
	for i in range(0, gb.shape[0]):
	    cam_mask[i, :, :] = mask

	cam_gb = np.multiply(cam_mask, gb)
	utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')
	#feature_extractor = FeatureExtractor(densenet.features, ['0'])
	#print(feature_extractor(Variable(x_test))[0][0].shape)
	#print(out)
