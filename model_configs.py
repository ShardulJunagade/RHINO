model_configs = [
    {   # Bihar to Bihar
        'train': 'Train Bihar',
        'test': 'Test Bihar',
        'config_file': 'configs-mine/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_bihar.py',
        'checkpoint_file': 'work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_bihar/epoch_36.pth',
        'val_dir': 'data/bihar/val',
        'inf_dir': 'results/train_bihar_test_bihar',
        'img_height': 640,
        'epoch': 36,
    },
    {
        # Haryana to Bihar
        'train': 'Train Haryana',
        'test': 'Test Bihar',
        'config_file': 'configs-mine/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_haryana.py',
        'checkpoint_file': 'work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_haryana/epoch_50.pth',
        'val_dir': 'data/haryana/val_bihar',
        'inf_dir': 'results/train_haryana_test_bihar',
        'img_height': 640,
        'epoch': 50,
    },
    {
        # Haryana to Bihar (epoch 45)
        'train': 'Train Haryana',
        'test': 'Test Bihar',
        'config_file': 'configs-mine/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_haryana.py',
        'checkpoint_file': 'work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_haryana/epoch_45.pth',
        'val_dir': 'data/haryana/val_bihar',
        'inf_dir': 'results/train_haryana_test_bihar_epoch_45',
        'img_height': 640,
        'epoch': 45,
    },
    {
        # m0 to m0
        'train': 'Train m0',
        'test': 'Test m0',
        'config_file': 'configs-mine/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_m0.py',
        'checkpoint_file': 'work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_m0/epoch_50.pth',
        'val_dir': 'data/m0/val',
        'inf_dir': 'results/train_m0_test_m0',
        'img_height': 640,
        'epoch': 50,
    },
    {
        # m0 to m0 (epoch 45)
        'train': 'Train m0',
        'test': 'Test m0',
        'config_file': 'configs-mine/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_m0.py',
        'checkpoint_file': 'work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_m0/epoch_45.pth',
        'val_dir': 'data/m0/val',
        'inf_dir': 'results/train_m0_test_m0_epoch_45',
        'img_height': 640,
        'epoch': 45,
    },
    {
        # SwinIR Bihar to Bihar
        'train': 'Train SwinIR Bihar',
        'test': 'Test SwinIR Bihar',
        'config_file': 'configs-mine/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_swinir_bihar_to_bihar.py',
        'checkpoint_file': 'work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_swinir_bihar_to_bihar/epoch_50.pth',
        'val_dir': 'data/swinir/test_bihar_same_class_count_10_120_1000_4x',
        'inf_dir': 'results/train_swinir_bihar_test_bihar',
        'img_height': 2560,
        'epoch': 50,
    },
    {
        # SwinIR Haryana to Bihar
        'train': 'Train SwinIR Haryana',
        'test': 'Test SwinIR Bihar',
        'config_file': 'configs-mine/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_swinir_haryana_to_bihar.py',
        'checkpoint_file': 'work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_swinir_haryana_to_bihar/epoch_50.pth',
        'val_dir': 'data/swinir/test_bihar_same_class_count_10_120_1000_4x',
        'inf_dir': 'results/train_swinir_haryana_test_bihar',
        'img_height': 2560,
        'epoch': 50,
    },
    {
        # SwinIR Haryana to Bihar (epoch 45)
        'train': 'Train SwinIR Haryana',
        'test': 'Test SwinIR Bihar',
        'config_file': 'configs-mine/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_swinir_haryana_to_bihar.py',
        'checkpoint_file': 'work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_swinir_haryana_to_bihar/epoch_45.pth',
        'val_dir': 'data/swinir/test_bihar_same_class_count_10_120_1000_4x',
        'inf_dir': 'results/train_swinir_haryana_test_bihar_epoch_45',
        'img_height': 2560,
        'epoch': 45,
    },
    {
        # SwinIR Haryana to Bihar (epoch 40)
        'train': 'Train SwinIR Haryana',
        'test': 'Test SwinIR Bihar',
        'config_file': 'configs-mine/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_swinir_haryana_to_bihar.py',
        'checkpoint_file': 'work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_swinir_haryana_to_bihar/epoch_40.pth',
        'val_dir': 'data/swinir/test_bihar_same_class_count_10_120_1000_4x',
        'inf_dir': 'results/train_swinir_haryana_test_bihar_epoch_40',
        'img_height': 2560,
        'epoch': 40,
    },
    {
        # SwinIR Haryana to Bihar (epoch 30)
        'train': 'Train SwinIR Haryana',
        'test': 'Test SwinIR Bihar',
        'config_file': 'configs-mine/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_swinir_haryana_to_bihar.py',
        'checkpoint_file': 'work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_swinir_haryana_to_bihar/epoch_30.pth',
        'val_dir': 'data/swinir/test_bihar_same_class_count_10_120_1000_4x',
        'inf_dir': 'results/train_swinir_haryana_test_bihar_epoch_30',
        'img_height': 2560,
        'epoch': 30,
    }
]