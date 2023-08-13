import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import dataloader, distributed
from tqdm import tqdm
from ultralytics.data import YOLODataset
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.augment import Format, v8_transforms, Compose, LetterBox
from ultralytics.data.utils import PIN_MEMORY, polygon2mask, img2label_paths, get_hash, HELP_URL
from ultralytics.utils import colorstr, RANK, LOCAL_RANK, TQDM_BAR_FORMAT, LOGGER


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask)
    ms = np.array(ms)
    return ms


class SAMFormat(Format):
    def __call__(self, labels):
        """Return formatted image, classes, bounding boxes & keypoints to be used by 'collate_fn'."""
        img = labels.pop('img')
        h, w = img.shape[:2]
        cls = labels.pop('cls')
        instances = labels.pop('instances')
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)

        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks).float()
            else:
                masks = torch.zeros(1 if self.mask_overlap else nl, img.shape[0] // self.mask_ratio,
                                    img.shape[1] // self.mask_ratio)
            labels['masks'] = masks
        if self.normalize:
            instances.normalize(w, h)
        labels['img'] = self._format_img(img)
        labels['cls'] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels['bboxes'] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        if self.return_keypoint:
            labels['keypoints'] = torch.from_numpy(instances.keypoints)
        # Then we can use collate_fn
        if self.batch_idx:
            labels['batch_idx'] = torch.zeros(nl)
        return labels

    def _format_segments(self, instances, cls, w, h):
        """convert polygon points to bitmap."""
        segments = instances.segments
        masks = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio)
        return masks, instances, cls

    def _format_img(self, img):
        """Format the image for YOLOv5 from Numpy array to PyTorch tensor."""
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = np.ascontiguousarray(img.transpose(2, 0, 1)[::-1])
        img = torch.from_numpy(img).div(255.0).float()
        return img


def collate_fn_yolo(batch):
    """Collates data samples into batches."""
    new_batch = {}
    keys = batch[0].keys()
    values = list(zip(*[list(b.values()) for b in batch]))
    for i, k in enumerate(keys):
        value = values[i]
        if k == 'img':
            value = torch.stack(value, 0)
        if k in ['masks', 'keypoints', 'bboxes', 'cls']:
            value = value
        new_batch[k] = value
    new_batch['batch_idx'] = list(new_batch['batch_idx'])
    for i in range(len(new_batch['batch_idx'])):
        new_batch['batch_idx'][i] += i  # add target image index for build_targets()
    new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
    return new_batch


class SAMYOLODataset(YOLODataset):
    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        try:
            import gc
            gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
            cache, exists = np.load(str(cache_path), allow_pickle=True).item(), True  # load dict
            gc.enable()
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=self.prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        if nf == 0:  # number of labels found
            raise FileNotFoundError(f'{self.prefix}No labels found in {cache_path}, can not start training. {HELP_URL}')

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels = cache['labels']
        self.im_files = [lb['im_file'] for lb in labels]  # update im_files

        def filter_label(label):
            if len(label['bboxes']) != len(label['segments']):
                label['segments'] = []
                label['bboxes'] = np.empty((0, 4))
                label['cls'] = np.empty((0, 1))
            return label

        labels = [filter_label(lb) for lb in labels]

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        if len_cls == 0:
            raise ValueError(f'All labels empty in {cache_path}, can not start training without labels. {HELP_URL}')
        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            SAMFormat(bbox_format='xyxy',
                   normalize=False,
                   return_mask=True,
                   return_keypoint=False,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms


class COCODataset(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        masks = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x + w, y + h])
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        if self.transform:
            image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        return image, torch.tensor(bboxes), torch.tensor(masks).float()


def collate_fn(batch):
    images, bboxes, masks = zip(*batch)
    images = torch.stack(images)
    return images, bboxes, masks


class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        # Adjust bounding boxes
        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

        return image, masks, bboxes


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = COCODataset(root_dir=cfg.dataset.train.root_dir,
                        annotation_file=cfg.dataset.train.annotation_file,
                        transform=transform)
    val = COCODataset(root_dir=cfg.dataset.val.root_dir,
                      annotation_file=cfg.dataset.val.annotation_file,
                      transform=transform)
    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                num_workers=cfg.num_workers,
                                collate_fn=collate_fn)
    return train_dataloader, val_dataloader


def build_yolo_dataset(yolo_cfg, img_path, batch, data, mode='train', rect=False, stride=32):
    """Build YOLO Dataset"""
    return SAMYOLODataset(
        img_path=img_path,
        imgsz=yolo_cfg.imgsz,
        batch_size=batch,
        augment=mode == 'train',  # augmentation
        hyp=yolo_cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=yolo_cfg.rect or rect,  # rectangular batches
        cache=yolo_cfg.cache or None,
        single_cls=yolo_cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=colorstr(f'{mode}: '),
        use_segments=True,
        classes=yolo_cfg.classes,
        data=data,
        fraction=yolo_cfg.fraction if mode == 'train' else 1.0)


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return DataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=collate_fn_yolo,
                              worker_init_fn=seed_worker,
                              generator=generator)


def load_yolo_datasets(yolo_cfg, data, batch, workers, stride=32):
    """Load YOLO Datasets"""
    if not isinstance(data["train"], list):
        data["train"] = [data["train"]]
    if not isinstance(data["val"], list):
        data["val"] = [data["val"]]
    train_path = [os.path.join(data["path"], tr) for tr in data["train"]]
    val_path = [os.path.join(data["path"], vl) for vl in data["val"]]
    train_dataset = build_yolo_dataset(yolo_cfg, train_path, batch, data, mode='train', stride=stride)
    val_dataset = build_yolo_dataset(yolo_cfg, val_path, batch, data, mode='val', stride=stride)
    train_dataloader = build_dataloader(train_dataset, batch, workers, shuffle=True, rank=-1)
    val_dataloader = build_dataloader(val_dataset, batch, workers, shuffle=False, rank=-1)
    return train_dataloader, val_dataloader
