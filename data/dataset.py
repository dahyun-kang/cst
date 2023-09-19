r""" Dataloader builder for few-shot classification and segmentation task """
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO


class FSCSDatasetModule(LightningDataModule): # LightningDataModule 상속받음
    """
    A LightningDataModule for FS-CS benchmark
    """
    def __init__(self, args, img_size=400):
        super().__init__()
        self.args = args
        self.datapath = args.datapath

        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]
        self.img_size = img_size
        self.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
        }
        self.transform = transforms.Compose([transforms.Resize(size=(self.img_size, self.img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.img_mean, self.img_std)])

        self.transforms_aug = []

    def train_dataloader(self): # PyTorch Lightning의 datamodule에서 train 데이터로더 오버라이딩
        dataset = self.datasets[self.args.benchmark](self.datapath,
                                                     fold=self.args.fold,
                                                     transform=self.transform,
                                                     transforms_aug=self.transforms_aug,
                                                     split='trn',
                                                     way=self.args.way,
                                                     shot=1)  # shot=1 fixed for training
        dataloader = DataLoader(dataset, batch_size=self.args.batchsize, shuffle=True, num_workers=8)
        return dataloader

    def val_dataloader(self): # PyTorch Lightning의 datamodule에서 val 데이터로더 오버라이딩
        dataset = self.datasets[self.args.benchmark](self.datapath,
                                                     fold=self.args.fold,
                                                     transform=self.transform,
                                                     split='val',
                                                     way=self.args.way,
                                                     shot=self.args.shot)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
        return dataloader

    def test_dataloader(self): # val이랑 같은 설정으로 테스트 로더 설정함
        return self.val_dataloader()
