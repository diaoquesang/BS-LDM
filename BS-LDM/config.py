from dataclasses import dataclass


@dataclass
class config():
    use_server = True
    test_epoch_interval = 10
    image_size = 1024

    # VQGAN
    ae_epoch_number = 1000
    ae_batch_size = 4
    milestones_g = [600]
    milestones_d = [600]
    initial_learning_rate_g = 1e-4
    initial_learning_rate_d = 5e-4


    # ldm
    batch_size = 4
    epoch_number = 2500
    initial_learning_rate = 2e-4
    milestones = [200, 400, 750, 1500]
    num_train_timesteps = 1000
    beta_start = 0.0008
    beta_end = 0.02
    beta_schedule = "squaredcos_cap_v2"
    offset_noise = True
    offset_noise_coefficient = 0.1
    output_feature_map = True
    clip_sample = True
    initial_clip_sample_range = 1.4
    clip_rate = 0.003
