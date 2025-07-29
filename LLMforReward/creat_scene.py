import yaml
import simpy
import random
from Environments.workShop import JobShop

def create_scene(root_dir, task, env_name, suffix):
    input_file = f"{root_dir}/env/{task}.yaml"
    output_file = f"{root_dir}/env/{task}{suffix}.yaml"
    with open(input_file, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)

    # Modify the "name" field
    data['name'] = f'{task}{suffix}'
    data['env']['env_name'] = f'{env_name}{suffix}'

    # Write the new YAML file
    with open(output_file, 'w') as new_yamlfile:
        yaml.safe_dump(data, new_yamlfile)   

    # Create training YAML file
    input_file = f"{root_dir}/train/{task}PPO.yaml"
    output_file = f"{root_dir}/train/{task}{suffix}PPO.yaml"

    with open(input_file, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)

    # Modify the "name" field
    data['params']['config']['name'] = data['params']['config']['name'].replace(task, f'{task}{suffix}')

    # Write the new YAML file
    with open(output_file, 'w') as new_yamlfile:
        yaml.safe_dump(data, new_yamlfile)

def make_env(args):
    env = simpy.Environment()
    seed = random.randint(0, 1000000)
    # Create a Workshop object
    sim = JobShop(env, args, seed)
    return sim, seed