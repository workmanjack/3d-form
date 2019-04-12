from data import DOWNLOADED_STLS_DIR, MODELNET10_DIR
from utils import PROJECT_ROOT
import os


def voxels_dim_ext(voxels_dim):
    return '' if voxels_dim == 32 else '_{}'.format(voxels_dim)


def stl_dir(obj):
    return os.path.join(DOWNLOADED_STLS_DIR, obj)


def modelnet_dir(category, dataset, obj):
    return os.path.join(MODELNET10_DIR, category, dataset, obj)


def shapenet_dir(path, voxels_dim):
    return os.path.join(PROJECT_ROOT, 'data/external/ShapeNetCore.v2', path,
                        'models/model_normalized{}.binvox'.format(voxels_dim_ext(voxels_dim)))


def good_recons_list(voxels_dim=32):
    return [
        # toilets
        modelnet_dir('toilet', 'test', 'toilet_0363_{}_x0_z0.binvox'.format(voxels_dim)),
        modelnet_dir('toilet', 'test', 'toilet_0397_{}_x0_z0.binvox'.format(voxels_dim)),
        # chairs (with legs!)
        modelnet_dir('chair', 'train', 'chair_0519_{}_x0_z0.binvox'.format(voxels_dim)),
        modelnet_dir('chair', 'train', 'chair_0408_{}_x0_z0.binvox'.format(voxels_dim)),
        modelnet_dir('chair', 'train', 'chair_0017_{}_x0_z0.binvox'.format(voxels_dim)),
        # sofa
        modelnet_dir('sofa', 'test', 'sofa_0757_{}_x0_z0.binvox'.format(voxels_dim)),
        # monitor
        modelnet_dir('monitor', 'test', 'monitor_0471_{}_x0_z0.binvox'.format(voxels_dim)),
        modelnet_dir('monitor', 'test', 'monitor_0531_{}_x0_z0.binvox'.format(voxels_dim)),
        # external
        stl_dir('CHIKEN{}.binvox'.format(voxels_dim_ext(voxels_dim))),
        stl_dir('chili{}.binvox'.format(voxels_dim_ext(voxels_dim))),
        stl_dir('apple{}.binvox'.format(voxels_dim_ext(voxels_dim))),
    ]


def bad_recons_list(voxels_dim=32):
    return [
        # not enough granularity - couldn't render well if we wanted to because objs are too small in 32x32x32
        stl_dir('R8_high_poly{}.binvox'.format(voxels_dim_ext(voxels_dim))),
        shapenet_dir('02691156/617993bd3425d570ca2bd098b9203af', voxels_dim),
        stl_dir('Bear{}.binvox'.format(voxels_dim_ext(voxels_dim))),
        stl_dir('Donkey{}.binvox'.format(voxels_dim_ext(voxels_dim))),
        # tough shape / sharp corners
        modelnet_dir('monitor', 'test', 'monitor_0496_{}_x0_z0.binvox'.format(voxels_dim)),
        modelnet_dir('monitor', 'train', 'monitor_0405_{}_x0_z0.binvox'.format(voxels_dim)),
        # boring shape
        modelnet_dir('monitor', 'test', 'monitor_0558_{}_x0_z0.binvox'.format(voxels_dim)),
        modelnet_dir('dresser', 'train', 'dresser_0010_{}_x0_z0.binvox'.format(voxels_dim)),
        modelnet_dir('sofa', 'train', 'sofa_0214_{}_x0_z0.binvox'.format(voxels_dim)),
        modelnet_dir('bed', 'test', 'bed_0558_{}_x0_z0.binvox'.format(voxels_dim)),
        # small gaps
        modelnet_dir('chair', 'train', 'chair_0118_{}_x0_z0.binvox'.format(voxels_dim)),
        modelnet_dir('chair', 'train', 'chair_0637_{}_x0_z0.binvox'.format(voxels_dim)),
        # bad voxelization
        modelnet_dir('chair', 'train', 'chair_0174_{}_x0_z0.binvox'.format(voxels_dim)),
        modelnet_dir('bed', 'train', 'bed_0156_{}_x0_z0.binvox'.format(voxels_dim))
    ]


def good_combos_list(voxels_dim=32):
    return [
        (stl_dir('CHIKEN{}.binvox'.format(voxels_dim_ext(voxels_dim))),
         modelnet_dir('chair', 'train', 'chair_0169_{}_x0_z0.binvox'.format(voxels_dim))),
        
        (stl_dir('CHIKEN{}.binvox'.format(voxels_dim_ext(voxels_dim))),
         modelnet_dir('toilet', 'test', 'toilet_0385_{}_x0_z0.binvox'.format(voxels_dim))),
        
        (modelnet_dir('toilet', 'test', 'toilet_0382_{}_x0_z0.binvox'.format(voxels_dim)),
         modelnet_dir('chair', 'train', 'chair_0645_{}_x0_z0.binvox'.format(voxels_dim))),
        
        (modelnet_dir('sofa', 'train', 'sofa_0051_{}_x0_z0.binvox'.format(voxels_dim)),
         modelnet_dir('toilet', 'test', 'toilet_0397_{}_x0_z0'.format(voxels_dim))),
        
        (modelnet_dir('chair', 'train', 'chair_0541_{}_x0_z0'.format(voxels_dim)),
         modelnet_dir('chair', 'train', 'chair_0637_{}_x0_z0'.format(voxels_dim))),
        
        (modelnet_dir('sofa', 'train', 'sofa_0573_{}_x0_z0'.format(voxels_dim)),
         modelnet_dir('chair', 'train', 'chair_0519_{}_x0_z0'.format(voxels_dim))),
        
        (stl_dir('CHIKEN{}.binvox'.format(voxels_dim_ext(voxels_dim))),
         stl_dir('911_high_poly{}.binvox'.format(voxels_dim_ext(voxels_dim)))),
        
         (stl_dir('CHIKEN{}.binvox'.format(voxels_dim_ext(voxels_dim))),
         stl_dir('RABBIT{}.binvox'.format(voxels_dim_ext(voxels_dim)))),
    ]


def bad_combos_list(voxels_dim=32):
    return [
        # not enough granularity
        (stl_dir('Bear{}.binvox'.format(voxels_dim_ext(voxels_dim))),
         stl_dir('911_high_poly{}.binvox'.format(voxels_dim_ext(voxels_dim)))),
        
        (stl_dir('R8_high_poly{}.binvox'.format(voxels_dim_ext(voxels_dim))),
         stl_dir('Lambo_high_poly{}.binvox'.format(voxels_dim_ext(voxels_dim)))),
        
        # too big of difference in object size
        (stl_dir('apple{}.binvox'.format(voxels_dim_ext(voxels_dim))),
         modelnet_dir('chair', 'train', 'chair_0017_{}_x0_z0'.format(voxels_dim))),
        
        (stl_dir('apple{}.binvox'.format(voxels_dim_ext(voxels_dim))),
         stl_dir('chili{}.binvox'.format(voxels_dim_ext(voxels_dim)))),
        
        # not aligned in cubic space
        (stl_dir('chili{}.binvox'.format(voxels_dim_ext(voxels_dim))),
         modelnet_dir('chair', 'train', 'chair_0169_{}_x0_z0'.format(voxels_dim)))
    ]
