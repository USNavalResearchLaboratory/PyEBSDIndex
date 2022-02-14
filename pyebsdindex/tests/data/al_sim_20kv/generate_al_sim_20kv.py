import kikuchipy as kp
import numpy as np
from orix.quaternion import Rotation
import skimage.io as skio


mp = kp.load(
    "/home/hakon/kode/emsoft/emdata/crystal_data/al/al_mc_mp_20kv.h5",
    projection="lambert",
    energy=20
)
rot = Rotation.identity()
detector = kp.detectors.EBSDDetector(
    shape=(100, 120),
    pc=(0.4, 0.6, 0.5),
    convention="tsl",
    sample_tilt=70,
    tilt=5.3,
)
s = mp.get_patterns(
    rotations=rot,
    detector=detector,
    compute=True,
    energy=20,
    dtype_out=np.uint8,
)
skio.imsave(
    "/home/hakon/kode/pyebsdindex/pyebsdindex/tests/data/al_sim_20kv/al_sim_20kv.png",
    s.data.squeeze()
)
