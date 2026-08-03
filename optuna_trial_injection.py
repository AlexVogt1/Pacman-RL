import optuna
import os

study_name = None

db_name = f"{study_name}.db"
# db_name = "pacman_optuna.db"

if os.name == 'nt':  # 'nt' means Windows
    # WINDOWS FIX: Force DB to live with the script to avoid path issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    db_path = os.path.join(script_dir, db_name)
else:
    # LINUX/CLUSTER: Do NOT change directory.
    # Assume the user submitted the job from a writable Scratch folder.
    # Just use the current working directory.
    db_path = os.path.join(os.getcwd(), db_name)

# Create URL (Handle slashes for SQLAlchemy)
db_path = db_path.replace("\\", "/")
storage_path = f"sqlite:///{db_path}"

net_arch_map = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[128, 128], vf=[128, 128])],
        "large": [dict(pi=[256, 256], vf=[256, 256])],
        "extra_large": [dict(pi=[512, 512], vf=[512, 512])],
        "XX-large": [dict(pi=[1024, 1024], vf=[512, 512])],
        "XXX-large": [dict(pi=[2048, 2048], vf=[512,512])],
        "extra_large_deep": [dict(pi=[512, 512, 256], vf=[512, 512, 256])],
        "huge": [dict(pi=[1024, 512, 256], vf=[1024, 512, 256])],
        "massive": [dict(pi=[2048, 1024, 512], vf=[2048, 1024, 512])]
    }
# net_arch = net_arch_map[net_arch_type]

# 2. Load the study pointer
study = optuna.load_study(
    study_name=study_name,
    storage=storage_path
)

# ReLU Studies

relu_study_32 = {
    "learning_rate": 0.00015029164216451537,
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 4,
    "gamma": 0.9849649840164338,
    "gae_lambda": 0.9478630648478699,
    "clip_range": 0.30146744857461144,
    "ent_coef": 0.0032283210141988753,
    "vf_coef": 0.898117285344109,
    "max_grad_norm": 1.3764813512787386,
    "net_arch": "extra_large_deep",
    "activation_fn": "relu"
}
relu_study_36 = {
    "learning_rate": 0.0002459867247824471,
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 3,
    "gamma": 0.9863544794378074,
    "gae_lambda": 0.9402424228270104,
    "clip_range": 0.2994936734451851,
    "ent_coef": 0.003607865860380944,
    "vf_coef": 0.954419209787857,
    "max_grad_norm": 1.089315751818332,
    "net_arch": "extra_large_deep",
    "activation_fn": "relu"
}

tanh_study_57 = {
    "learning_rate": 5.203809987497978e-05,
    "n_steps": 1024,
    "batch_size": 512,
    "n_epochs": 14,
    "gamma": 0.9917561021672321,
    "gae_lambda": 0.9354787795469003,
    "clip_range": 0.31678566349099524,
    "ent_coef": 0.009088484161261507,
    "vf_coef": 0.7636689782785554,
    "max_grad_norm": 3.017003521831221,
    "net_arch": "extra_large_deep",
    "activation_fn": "tanh"
}

tanh_study_32 = {
    "learning_rate": 0.00014557736432668238,
    "n_steps": 1024,
    "batch_size": 1024,
    "n_epochs": 9,
    "gamma": 0.9923280352994776,
    "gae_lambda": 0.9460508746472488,
    "clip_range": 0.29808777922159924,
    "ent_coef": 0.005470468863837204,
    "vf_coef": 0.7231579980921344,
    "max_grad_norm": 3.7373772239923007,
    "net_arch": "extra_large_deep",
    "activation_fn": "tanh"
}

## Adjusted trials
huge_tanh_study_89 = {
    "learning_rate": 0.00011175929723153039,
    "n_steps": 1024,
    "batch_size": 1024,
    "n_epochs": 9,
    "gamma": 0.9985792029291427,
    "gae_lambda": 0.9626357286059191,
    "clip_range": 0.2536307691065866,
    "ent_coef": 0.00016052828614421442,
    "vf_coef": 0.817852565473667,
    "max_grad_norm": 3.2486899825832425,
    "net_arch": "huge",
    "activation_fn": "tanh"
}

ng_bp_bs_trial_109 = {
    "learning_rate": 0.0001463056419756806,
    "n_steps": 1024,
    "batch_size": 512,
    "n_epochs": 7,
    "gamma": 0.9981470149374636,
    "gae_lambda": 0.9465730424011484,
    "clip_range": 0.21144869867198185,
    "ent_coef": 0.006720984926240852,
    "vf_coef": 0.7150442916544829,
    "max_grad_norm": 3.4811827631898695,
    "net_arch": "extra_large_deep",
    "activation_fn": "tanh"
}

trial_2271 = {
    "learning_rate": 0.00013359399511614218,
    "n_steps": 1024,
    "batch_size": 1024,
    "n_epochs": 15,
    "gamma": 0.9998508603095597,
    "gae_lambda": 0.948412533791077,
    "clip_range": 0.12494103694930075,
    "ent_coef": 0.0030916541412902288,
    "vf_coef": 0.9671319737792027,
    "max_grad_norm": 4.97093755400307,
    "net_arch": "extra_large",
    "activation_fn": "leakyRelu"
}

trial_881 = {
    "learning_rate": 0.00010316513909684482,
    "n_steps": 1024,
    "batch_size": 1024,
    "n_epochs": 10,
    "gamma": 0.9995297515683058,
    "gae_lambda": 0.9140449449454066,
    "clip_range": 0.1392124932393117,
    "ent_coef": 1.1324255425365424e-05,
    "vf_coef": 0.9080387604621629,
    "max_grad_norm": 2.3626981532172966,
    "net_arch": "extra_large",
    "activation_fn": "leakyRelu"
}

trial_1394 = {
    "learning_rate": 0.00018208198315112391,
    "n_steps": 1024,
    "batch_size": 1024,
    "n_epochs": 12,
    "gamma": 0.9995907115144507,
    "gae_lambda": 0.9410474136412406,
    "clip_range": 0.11504649465443167,
    "ent_coef": 0.0010804914744171998,
    "vf_coef": 0.8617335443729136,
    "max_grad_norm": 1.6561988502834568 ,
    "net_arch": "large",
    "activation_fn": "mish"
}

trial_1312 = {
    "learning_rate": 0.0001172042893925917,
    "n_steps": 1024,
    "batch_size": 1024,
    "n_epochs": 12,
    "gamma": 0.9994483284370438,
    "gae_lambda": 0.9414162020120727,
    "clip_range": 0.11428788725398678,
    "ent_coef": 0.011032883443994524,
    "vf_coef": 0.8584200956488066,
    "max_grad_norm": 0.30026707502881256,
    "net_arch": "extra_large",
    "activation_fn": "leakyRelu"
}



# 4. Push it to the Database
study.enqueue_trial(relu_study_32)
study.enqueue_trial(relu_study_36)
study.enqueue_trial(tanh_study_57)
study.enqueue_trial(tanh_study_32)

print("Instruction sent! The next available worker will run this config.")