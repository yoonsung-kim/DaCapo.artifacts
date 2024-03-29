from .simulate import simulation

def pick_configs(config_set_list, time_limit):
    best_acc = -0.1
    best_config_set = None
    
    for config_set in config_set_list:
        tmp_acc = simulation(config_set, time_limit)
        if tmp_acc > best_acc:
            best_acc = tmp_acc
            best_config_set = config_set
    
    return best_config_set

def thief_scheduler(config_set_list, time_limit):
    best_config_set = pick_configs(config_set_list, time_limit)
    
    # no thief scheduler, cause Orin AGX does not support MPS
    
    return best_config_set