# Trained behaviour Models

## Base models
- All base models were wrapped with `env = wrap_env(env, skip=4, wrap_reward='normalise') `
- 

## Specific behavlets
- All used the build base_0
- config details are in the `collect_data.py` in the pacman_cfgs dict

## General Behaviours

### Aggression 
build used: single_life_ep
```
env = wrap_env(env, skip=4, wrap_reward='normalise', step_reward=-1.225, aggression=True) 
```

### Speed
build used: single_life_ep
```
env = wrap_env(env, skip=4, wrap_reward='normalise', step_reward=-1.225)
``` 