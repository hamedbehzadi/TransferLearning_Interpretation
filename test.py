# from uutils.torch_uu import l2_sim_torch
# import numpy as np
# import torch
# import config as cfg
# from receptive_field import Receptive_Field

# print(cfg.BASE_MODEL)


# out1 = torch.from_numpy(np.ones((3,3), dtype=np.float64))
# out2 = torch.from_numpy(np.zeros((3,3), dtype=np.float64))

# print(out1)

# print(l2_sim_torch(out1, out2, sim_type='op_torch'))

# print(type(cfg.BASE_MODEL.features.denseblock1))
# rf = Receptive_Field(cfg.BASE_MODEL)
# print(rf.layer_map)

import generate_visualisations
import image_isolator

_override = True


if not _override:
    print('yes')

#generate_visualisations.run()
#image_isolator.run()
