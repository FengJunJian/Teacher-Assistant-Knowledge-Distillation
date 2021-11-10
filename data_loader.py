import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from copy import copy
NUM_WORKERS = 2

ship_classes=[
'vessel',
 'Boat',
 'bulk_cargo_carrier',
 'Buoy',
 'container_ship',
 'Ferry',
 'fishing_boat',
 'general_cargo_ship',
 'Kayak',
 'ore_carrier',
 'Other',
 'passenger_ship',
 'Sail_boat',
 'Speed_boat'
]


def get_cifar(num_classes=100, dataset_dir='./data', batch_size=128, crop=False):
	"""
	:param num_classes: 10 for cifar10, 100 for cifar100
	:param dataset_dir: location of datasets, default is a directory named 'data'
	:param batch_size: batchsize, default to 128
	:param crop: whether or not use randomized horizontal crop, default to False
	:return:
	"""
	normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
	simple_transform = transforms.Compose([transforms.ToTensor(), normalize])
	
	if crop is True:
		train_transform = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize
		])
	else:
		train_transform = simple_transform
	
	if num_classes == 100:
		trainset = torchvision.datasets.CIFAR100(root=dataset_dir, train=True,
												 download=True, transform=train_transform)
		
		testset = torchvision.datasets.CIFAR100(root=dataset_dir, train=False,
												download=True, transform=simple_transform)
	else:
		trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True,
												 download=True, transform=train_transform)
		
		testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
												download=True, transform=simple_transform)
		
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=NUM_WORKERS,
											  pin_memory=True, shuffle=True)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=NUM_WORKERS,
											 pin_memory=True, shuffle=False)
	return trainloader, testloader


def get_ship(num_classes=14, dataset_dir='./data', batch_size=128, crop=True):
	"""
	:param num_classes: 10 for cifar10, 100 for cifar100, 14 for ship
	:param dataset_dir: location of datasets, default is a directory named 'data'
	:param batch_size: batchsize, default to 128
	:param crop: whether or not use randomized horizontal crop, default to False
	:return:
	"""

	normalize = transforms.Normalize(mean=(0.51264606, 0.55715489, 0.6386575), std=(0.15772002, 0.14560729, 0.13691749))
	simple_transform = transforms.Compose([transforms.Resize((32, 64)),transforms.RandomCrop(32, padding=4),
										   transforms.ToTensor(), normalize])

	if crop is True:
		train_transform = transforms.Compose([
			transforms.Resize((32, 64)),  # (h,w) (64,128)
			transforms.RandomCrop(32, padding=4),#(32)
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize
		])
	else:
		train_transform = simple_transform

	dataset = torchvision.datasets.ImageFolder(dataset_dir)  # transform=transforms.ToTensor()
	N = len(dataset)
	train_size = int(N * 0.8)
	val_size = N - train_size

	trainDataset, testDataset = torch.utils.data.random_split(dataset, [train_size, val_size])
	trainDataset.dataset = copy(dataset)

	trainDataset.dataset.transform = train_transform

	#val_dataset = testDataset
	testDataset.dataset.transform = simple_transform


	trainloader = DataLoader(trainDataset, batch_size=batch_size, num_workers=NUM_WORKERS,
											  pin_memory=True, shuffle=True)
	testloader = DataLoader(testDataset, batch_size=batch_size, num_workers=NUM_WORKERS,
											 pin_memory=True, shuffle=False)
	return trainloader, testloader


if __name__ == "__main__":
	print("SHIP")
	trainloader, testloader=get_ship(dataset_dir='../Classification_advanced')
	iter=trainloader.__iter__()
	iter1 = testloader.__iter__()
	print("---" * 20)
	# print("CIFAR10")
	# print(get_cifar(10))
	# print("---"*20)
	# print("---"*20)
	# print("CIFAR100")
	# print(get_cifar(100))
