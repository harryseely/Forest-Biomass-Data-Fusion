# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import ocnn
from ocnn.octree import Octree, Points


class CustomTransform:
    r''' [CUSTOMIZED BY ME FOR BIOMASS DL] A boilerplate class which transforms an input data for :obj:`ocnn`.
    The input data is first converted to :class:`Points`, then randomly transformed
    (if enabled), and converted to an :class:`Octree`.

    Args:
      depth (int): The octree depth.
      full_depth (int): The octree layers with a depth small than
          :attr:`full_depth` are forced to be full.
      use_normals (bool): Whether to use surface normals
      orient_normal (bool): Orient point normals along the specified axis, which is
          useful when normals are not oriented.
      augment (bool): If true, performs the data augmentation.
      angle (list): A list of 3 float values to generate random rotation angles.
      interval (list): A list of 3 float values to represent the interval of
        rotation angles.
      jittor (float): The maximum jitter values.
    '''

    def __init__(self, depth: int,
                 full_depth: int,
                 use_normals=False,
                 augment=False,
                 angle=(0, 0, 5),  # x,y,z axes (currently a small rotation around z axis)
                 interval=(1, 1, 1),
                 jitter=0.125,
                 **kwargs):

        super().__init__()

        # for octree building
        self.use_normals = use_normals
        self.depth = depth
        self.full_depth = full_depth

        # for data augmentation
        self.augment = augment
        self.angle = angle
        self.interval = interval
        self.jitter = jitter

    def __call__(self, sample: dict, idx: int):
        r''''''

        points = self.preprocess(sample, idx)
        output = self.transform(points, idx)
        output['octree'] = self.points2octree(output['points'])
        return output

    def preprocess(self, sample: dict, idx: int):
        r''' Transforms :attr:`sample` to :class:`Points` and performs some specific
        transformations, like normalization.
        '''

        # Select coordinates
        xyz = torch.from_numpy(sample['points']).float()
        # Convert features to tensor (if they are available)
        if sample['features'] is not None:
            features = torch.from_numpy(sample['features']).float()
        else:
            features = None
        # Select normals
        if self.use_normals:
            normals = torch.from_numpy(sample['normals']).float()
        else:
            normals = None

        # Convert to points object that is compatible with octree
        points = Points(xyz, normals=normals, features=features)

        # Need to normalize the point cloud into one unit sphere in [-0.8, 0.8]
        bbmin, bbmax = points.bbox()
        points.normalize(bbmin, bbmax, scale=0.8)
        points.scale(torch.Tensor([0.8, 0.8, 0.8]))

        return points

    def transform(self, points: Points, idx: int):
        r''' Applies the general transformations provided by :obj:`ocnn`.
        '''

        # The augmentations including rotation, scaling, and jittering.
        if self.augment:
            rng_angle, rng_jitter = self.rnd_parameters()
            points.rotate(rng_angle)
            points.translate(rng_jitter)

        # Orient the point normals along a given axis
        points.orient_normal(axis="x")

        # !!! NOTE: Clip the point cloud to [-1, 1] before building the octree
        inbox_mask = points.clip(min=-1, max=1)
        return {'points': points, 'inbox_mask': inbox_mask}

    def points2octree(self, points: Points):
        r''' Converts the input :attr:`points` to an octree.
        '''

        octree = Octree(self.depth, self.full_depth)
        octree.build_octree(points)
        return octree

    def rnd_parameters(self):
        r''' Generates random parameters for data augmentation.
        '''

        rnd_angle = [None] * 3
        for i in range(3):
            rot_num = self.angle[i] // self.interval[i]
            rnd = torch.randint(low=-rot_num, high=rot_num + 1, size=(1,))
            rnd_angle[i] = rnd * self.interval[i] * (3.14159265 / 180.0)

        rnd_angle = torch.cat(rnd_angle)

        rnd_jitter = torch.rand(3) * (2 * self.jitter) - self.jitter

        return rnd_angle, rnd_jitter


class CustomCollateBatch:
    r""" Merge a list of octrees and points into a batch.
  """

    def __init__(self, batch_size: int, merge_points: bool = False):
        self.merge_points = merge_points
        self.batch_size = batch_size

    def __call__(self, batch: list):
        assert type(batch) == list

        outputs = {}
        for key in batch[0].keys():
            outputs[key] = [b[key] for b in batch]

            # Merge a batch of octrees into one super octree
            if 'octree' in key:
                octree = ocnn.octree.merge_octrees(outputs[key])
                # NOTE: remember to construct the neighbor indices
                octree.construct_all_neigh()
                outputs[key] = octree

            # Merge a batch of points
            if 'points' in key and self.merge_points:
                outputs[key] = ocnn.octree.merge_points(outputs[key])

            # Convert the labels to a Tensor
            if 'target' in key:
                num_samples = len(outputs['target'])
                num_targets = outputs['target'][0].shape[0]
                target_reshape = torch.cat(outputs['target'])
                target_reshape = torch.reshape(target_reshape, (num_samples, num_targets))
                outputs['target'] = target_reshape

            # Merge list of terrain rasters into single tensor
            if 'terrain' in key:
                outputs['terrain'] = torch.stack(outputs['terrain'])

            # Convert the spectral list of tensors to a single Tensor of correct shape
            if 'spectral' in key:
                spectral_tensor = torch.cat(outputs['spectral'])
                batch_size, num_spectra = spectral_tensor.shape
                outputs['spectral'] = spectral_tensor.reshape(batch_size, 1, num_spectra)


        return outputs
