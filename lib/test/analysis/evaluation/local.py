from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/sqh/lihao/AnyTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/sqh/lihao/AnyTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/sqh/lihao/AnyTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/home/sqh/lihao/AnyTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/sqh/lihao/AnyTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/sqh/lihao/AnyTrack/data/lasot'
    settings.network_path = '/home/sqh/lihao/AnyTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/sqh/lihao/AnyTrack/data/nfs'
    settings.otb_path = '/home/sqh/lihao/AnyTrack/data/otb'
    settings.prj_dir = '/home/sqh/lihao/AnyTrack'
    settings.result_plot_path = '/home/sqh/lihao/AnyTrack/output/test/result_plots'
    settings.results_path = '/home/sqh/lihao/AnyTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/sqh/lihao/AnyTrack/output'
    settings.segmentation_path = '/home/sqh/lihao/AnyTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/sqh/lihao/AnyTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/sqh/lihao/AnyTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/sqh/lihao/AnyTrack/data/trackingnet'
    settings.uav_path = '/home/sqh/lihao/AnyTrack/data/uav'
    settings.vot18_path = '/home/sqh/lihao/AnyTrack/data/vot2018'
    settings.vot22_path = '/home/sqh/lihao/AnyTrack/data/vot2022'
    settings.vot_path = '/home/sqh/lihao/AnyTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

