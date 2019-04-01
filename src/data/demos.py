from data import DOWNLOADED_STLS_DIR, MODELNET10_DIR


def stl_dir(obj):
    return os.path.join(DOWNLOADED_STLS_DIR, obj)


def modelnet_dir(category, dataset, obj):
    return os.path.join(MODELNET10_DIR, category, dataset, obj)


def shapenet_dir(path):
    os.path.join(PROJECT_ROOT, 'data/external/ShapeNetCore.v2', 'models/model_normalized.binvox')


GOOD_RECONS = [
    # toilets
    modelnet_dir('toilet', 'test', 'toilet_0363_32_x0_z0.binvox'),
    modelnet_dir('toilet', 'test', 'toilet_0397_32_x0_z0.binvox'),
    # chairs (with legs!)
    modelnet_dir('chair', 'train', 'chair_0519_32_x0_z0.binvox'),
    modelnet_dir('chair', 'train', 'chair_0408_32_x0_z0.binvox'),
    modelnet_dir('chair', 'train', 'chair_0017_32_x0_z0.binvox'),
    # sofa
    modelnet_dir('sofa', 'test', 'sofa_0757_32_x0_z0.binvox'),
    # monitor
    modelnet_dir('monitor', 'train', 'monitor_0471_32_x0_z0.binvox'),
    modelnet_dir('monitor', 'test', 'monitor_0531_32_x0_z0.binvox')
    # external
    stl_dir('CHIKEN.binvox'),
    stl_dir('chili.binvox'),
    stl_dir('apple.binvox'),
]

BAD_RECONS = [
    # not enough granularity - couldn't render well if we wanted to because objs are too small in 32x32x32
    stl_dir('R8_high_poly.binvox'),
    shapenet_dir('02691156/617993bd3425d570ca2bd098b9203af'),
    stl_dir('Bear.binvox'),
    stl_dir('Donkey.binvox'),
    # tough shape / sharp corners
    modelnet10_dir('monitor', 'test', 'monitor_0496_32_x0_z0.binvox'),
    modelnet10_dir('monitor', 'train', 'monitor_0405_32_x0_z0.binvox'),
    # boring shape
    modelnet10_dir('monitor', 'train', 'monitor_0558_32_x0_z0.binvox'),
    modelnet10_dir('dresser', 'train', 'dresser_0010_32_x0_z0.binvox'),
    modelnet10_dir('sofa', 'train', 'sofa_0214_32_x0_z0.binvox'),
    modelnet10_dir('bed', 'train', 'bed_0558_32_x0_z0.binvox'),
    # small gaps
    modelnet10_dir('chair', 'train', 'chair_0118_32_x0_z0.binvox'),
    modelnet10_dir('chair', 'train', 'chair_0637_32_x0_z0.binvox'),
    # bad voxelization
    modelnet10_dir('chair', 'train', 'chair_0174_32_x0_z0.binvox'),
    modelnet10_dir('bed', 'train', 'bed_0156_32_x0_z0.binvox')
]

GOOD_COMBOS = [
    (stl_dir('CHIKEN.binvox'), modelnet_dir('chair', 'train', 'chair_0169_32_x0_z0.binvox')),
    (stl_dir('CHIKEN.binvox'), modelnet_dir('toilet', 'train', 'toilet_0385_32_x0_z0.binvox')),
    (modelnet_dir('toilet', 'train', 'toilet_0382_32_x0_z0.binvox'), modelnet_dir('chair', 'train', 'chair_0645_32_x0_z0.binvox')),
    (modelnet_dir('sofa', 'train', 'sofa_0051_32_x0_z0.binvox'), modelnet_dir('toilet', 'train', 'toilet_0397_32_x0_z0')),
    (modelnet_dir('chair', 'train', 'chair_0541_32_x0_z0'), modelnet_dir('chair', 'train', 'chair_0637_32_x0_z0')),
    (modelnet_dir('sofa', 'train', 'sofa_0573_32_x0_z0'), modelnet_dir('chair', 'train', 'chair_0519_32_x0_z0')),
    (stl_dir('CHIKEN.binvox'), stl_dir('911_high_poly.binvox')),
    (stl_dir('CHIKEN.binvox'), stl_dir('RABBIT.binvox')),
]

BAD_COMBOS = [
    # not enough granularity
    (stl_dir('Bear.binvox'), stl_dir('911_high_poly.binvox')),
    (stl_dir('R8_high_poly.binvox'), stl_dir('Lambo_high_poly.binvox')),
    # too big of difference in object size
    (stl_dir('apple'), modelnet_dir('chair', 'train', 'chair_0017_32_x0_z0')),
    (stl_dir('apple'), stl_dir('chili')),
    # not aligned in cubic space
    (stl_dir('chili'), modelnet_dir('chair', 'train', 'chair_0169_32_x0_z0')),
]
