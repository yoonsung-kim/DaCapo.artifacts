from scipy.stats import gmean
from scipy.optimize import curve_fit

import numpy as np

END_EPOCH = 30
END_TARGET_ACC = 0.95
MICROPROFILE_EXPECTATION_FACTOR = 0.95

def scipy_fit(func, xp, yp, sigma=None):
    popt, _ = curve_fit(func, xp, yp, sigma=sigma, method='dogbox', absolute_sigma=True)
    return lambda x: func(x, *popt)

def curve(x, b0, b1, b2):
    return 1 - (1 / (b0 * x + b1) + b2)

def get_expected_accuracy(tr_config, profile_res):
    before_test_acc = profile_res["before_train_acc"]
    after_test_acc = profile_res["after_train_acc"]
    microprofile_epochs = tr_config["microprofile_epochs"]
    
    seed_x = np.array([0, END_EPOCH])
    seed_y = np.array([before_test_acc, END_TARGET_ACC])
    seed_curve = scipy_fit(curve, seed_x, seed_y)
    
    microprofile_expected_values = seed_curve(microprofile_epochs)
    microprofile_deviation = min(after_test_acc / (microprofile_expected_values * MICROPROFILE_EXPECTATION_FACTOR), 1)
    new_end_acc = END_TARGET_ACC * microprofile_deviation
    
    new_seed_y = np.array([before_test_acc, new_end_acc])
    seed_curve = scipy_fit(curve, seed_x, new_seed_y)

    booster_pts_x = np.linspace(min(seed_x), max(seed_x), 4)
    booster_pts_y = seed_curve(booster_pts_x)

    xp = np.concatenate([booster_pts_x, seed_x, [microprofile_epochs,]])
    yp = np.concatenate([booster_pts_y, new_seed_y, [after_test_acc,]])
    
    try:
        fn = scipy_fit(curve, xp, yp)
        return fn(tr_config["epochs"])
    except:
        return after_test_acc
    
def simulation(config_set, time_limit):
    inference_config = config_set["inference"]
    train_config = config_set["train"]
    label_config = config_set["label"]
    profile_res = config_set["profile"]
    
    inference_resource = inference_config["resource"]
    train_resource = train_config["resource"]
    label_resource = label_config["resource"]
    
    num_inputs = time_limit * inference_config["FPS"]
    
    num_train_inputs = num_inputs * train_config["sample_rate"]
    num_batches = (num_train_inputs + train_config["train_batch_size"] - 1) // train_config["train_batch_size"]
    
    measured_time = inference_resource * num_inputs
    measured_time += label_resource * num_train_inputs
    measured_time += train_resource * num_batches * train_config["epochs"]
    
    if measured_time > time_limit: return 0
    
    return get_expected_accuracy(train_config, profile_res)