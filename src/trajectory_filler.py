import torch
import lietorch
from lietorch import SE3
from src.factor_graph import FactorGraph
from tqdm import tqdm
from src.utils.datasets import BaseDataset
from src.utils.Printer import FontColor
from src.utils.mono_priors.img_feature_extractors import predict_img_features, get_feature_extractor

class PoseTrajectoryFiller:
    """ This class is used to fill in non-keyframe poses 
        mainly inherited from DROID-SLAM
    """
    def __init__(self, cfg, net, video, printer, device='cuda:0'):
        self.cfg = cfg

        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.count = 0
        self.video = video
        self.device = device
        self.printer = printer

        # mean, std for image normalization
        self.MEAN = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.STDV = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

        self.uncertainty_aware = cfg['tracking']["uncertainty_params"]['activate']        
        
    def setup_feature_extractor(self):
        if self.uncertainty_aware:
            self.feat_extractor = get_feature_extractor(self.cfg)

    @torch.amp.autocast('cuda',enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image)

    def __fill(self, timestamps, images, depths, intrinsics, dino_features):
        """ fill operator """
        tt = torch.tensor(timestamps, device=self.device)
        images = torch.stack(images, dim=0)
        if depths is not None:
            depths = torch.stack(depths, dim=0)
        intrinsics = torch.stack(intrinsics, 0)
        if dino_features is not None:
            dino_features = torch.stack(dino_features, dim=0).to(self.device)
        inputs = images.to(self.device)

        ### linear pose interpolation ###
        N = self.video.counter.value
        M = len(timestamps)

        ts = self.video.timestamp[:N]
        Ps = SE3(self.video.poses[:N])

        # found the location of current timestamp in keyframe queue
        t0 = torch.tensor([ts[ts<=t].shape[0] - 1 for t in timestamps])
        t1 = torch.where(t0 < N-1, t0+1, t0)

        # time interval between nearby keyframes
        dt = ts[t1] - ts[t0] + 1e-3
        dP = Ps[t1] * Ps[t0].inv()

        v = dP.log() / dt.unsqueeze(dim=-1)
        w = v * (tt - ts[t0]).unsqueeze(dim=-1)
        Gs = SE3.exp(w) * Ps[t0]

        # extract features (no need for context features)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        fmap = self.__feature_encoder(inputs)

        # temporally put the non-keyframe at the end of keyframe queue
        self.video.counter.value += M
        self.video[N:N+M] = (tt, images[:, 0], Gs.data, 1, depths, intrinsics / 8.0, fmap, None, None, dino_features)

        if self.uncertainty_aware:
            self.video.update_uncertainty_mask_given_index(range(N,N+M))

        graph = FactorGraph(self.video, self.update)
        # build edge between current frame and nearby keyframes for optimization
        graph.add_factors(t0.cuda(), torch.arange(N, N+M).cuda())
        graph.add_factors(t1.cuda(), torch.arange(N, N+M).cuda())

        for _ in range(12):
            graph.update(N, N+M, motion_only=True)

        Gs = SE3(self.video.poses[N:N+M].clone())
        self.video.counter.value -= M

        return [Gs]

    @torch.no_grad()
    def __call__(self, image_stream:BaseDataset):
        """ fill in poses of non-keyframe images. """

        # store all camera poses
        pose_list = []
        dino_feats = []

        timestamps = []
        images = []
        intrinsics = []
        dino_features = []

        self.printer.print("Filling full trajectory ...",FontColor.INFO)
        intrinsic = image_stream.get_intrinsic()

        # **** qingshufan modified code start ****
        
        for i in tqdm(range(len(image_stream))):
            timestamp, image, _, _  = image_stream[i]
            
        # **** qingshufan modified code start ****

            timestamps.append(timestamp)
            images.append(image)
            intrinsics.append(intrinsic)
            if self.uncertainty_aware:
                dino_feature = predict_img_features(self.feat_extractor,
                                                    timestamp,image,
                                                    self.cfg,
                                                    self.device,
                                                    save_feat=False)
                dino_features.append(dino_feature)
            else:
                dino_features = None

            if len(timestamps) == 16:
                pose_list += self.__fill(timestamps, images, None, intrinsics, dino_features)
                dino_feats += dino_features
                timestamps, images, intrinsics, dino_features = [], [], [], []

        if len(timestamps) > 0:
            pose_list += self.__fill(timestamps, images, None, intrinsics, dino_features)
            dino_feats += dino_features

        # stitch pose segments together
        return lietorch.cat(pose_list, dim=0), dino_feats