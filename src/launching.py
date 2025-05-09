from paths import CONFIG_DIR

use_slurm = False
if use_slurm:
    raise NotImplementedError("slurm mode not implemented")
yml_configs = [ff for ff in CONFIG_DIR.iter() if ff.endswith(".yml")]
import pdb; pdb.set_trace()
