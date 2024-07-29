import torch.utils.data as torchdata
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from TransferLearningInterpretation import config as cfg


class Data(object):
    def __init__(self):
        if cfg._settings["model"] == 4:
            config = resolve_data_config({}, model=cfg.BASE_MODEL)
            self.data_transform = create_transform(**config)
        else:
            self.data_transform = transforms.Compose([
                transforms.Resize(cfg.INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.data_augment_transform = transforms.Compose([
                transforms.Resize(cfg.INPUT_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(60),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

class DataSingle(Data):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.model_dataset = datasets.ImageFolder(cfg.DATA_PATH, transform=self.data_transform)
        total_count = len(self.model_dataset.imgs)
        train_count = int(0.8 * total_count)
        test_count = total_count - train_count
        self.train_dataset, self.test_dataset = torchdata.random_split(
            self.model_dataset, (train_count, test_count), generator=torch.Generator().manual_seed(42))

        self.train_loader = torchdata.DataLoader(
            self.train_dataset, batch_size=cfg.TRAINING_BATCHSIZE, shuffle=True)
        self.test_loader = torchdata.DataLoader(
            self.test_dataset, batch_size=cfg.VALIDATION_BATCHSIZE, shuffle=False)

    def get_filename(self, index: int) -> str:
        indice = self.test_dataset.indices[index]
        return self.model_dataset.samples[indice][0]


class DataSplit(Data):
    def __init__(self, train_root, test_root, cfg):
        super().__init__()
        self.train_dataset = datasets.ImageFolder(train_root, transform=self.data_transform)
        self.test_dataset = datasets.ImageFolder(test_root, transform=self.data_transform)

        self.train_loader = torchdata.DataLoader(
            self.train_dataset, batch_size=cfg.TRAINING_BATCHSIZE, shuffle=True)
        self.test_loader = torchdata.DataLoader(
            self.test_dataset, batch_size=cfg.VALIDATION_BATCHSIZE, shuffle=False)

    def get_filename(self, index: int) -> str:
        assert abs(index) < len(self.test_dataset.imgs)
        return self.test_dataset.imgs[index]


class CustomActivationData(Data):
    def __init__(self, root: str) -> None:
        super().__init__()
        self.rf_dataset = datasets.ImageFolder(root, transform=self.data_transform)
        self.rf_loader = torchdata.DataLoader(self.rf_dataset)

    def get_filename(self, index: int) -> str:
        assert abs(index) < len(self.rf_dataset.imgs)
        return self.rf_dataset.imgs[index]

class ImagNet():
    def __init__(self,args,cfg):
        from torchvision import models
        from InterpretationViaModelInversion.lmdbDataset import LMDBDataset
        transformation = models.VGG19_Weights.IMAGENET1K_V1.transforms()
        dataset = LMDBDataset('./datasets_antwerp/ImageNet/train.lmdb', transform=transformation)
        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAINING_BATCHSIZE, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)
        val_dataset_dir = datasets.ImageFolder('./project_antwerp/dataset/ImageNet/validation_category/',
                                               transformation)
        self.test_loader = torch.utils.data.DataLoader(val_dataset_dir, batch_size=256, shuffle=False)