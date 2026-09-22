import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from lib.test.vot.anytrack_class import run_vot_exp



run_vot_exp('anytrack', 'anytrack', vis=False, out_conf=True, channel_type='rgbd', run_id=20, conf_thr=0.75, update_intervals=50, modalities=['rgb','depth'], fill_modalities=['text,audio'], miss=False)
