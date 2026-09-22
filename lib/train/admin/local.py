class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/sqh/lihao/AnyTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/sqh/lihao/AnyTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/sqh/lihao/AnyTrack/pretrained_networks'
        self.got10k_val_dir = '/home/sqh/lihao/AnyTrack/data/got10k/val'
        self.lasot_lmdb_dir = '/home/sqh/lihao/AnyTrack/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/sqh/lihao/AnyTrack/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/home/sqh/lihao/AnyTrack/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/home/sqh/lihao/AnyTrack/data/coco_lmdb'
        self.coco_dir = '/home/sqh/lihao/AnyTrack/data/coco'
        self.lasot_dir = '/home/sqh/lihao/AnyTrack/data/lasot'
        self.got10k_dir = '/home/sqh/lihao/AnyTrack/data/got10k/train'
        self.trackingnet_dir = '/home/sqh/lihao/AnyTrack/data/trackingnet'
        self.depthtrack_dir = '/home/sqh/lihao/AnyTrack/data/depthtrack/train'
        self.lasher_dir = '/home/sqh/lihao/AnyTrack/data/lasher/trainingset'
        self.visevent_dir = '/home/sqh/lihao/AnyTrack/data/visevent/train'
        self.rgbdt_dir = '/home/sqh/lihao/AnyTrack/data/rgbdt500/Train'
