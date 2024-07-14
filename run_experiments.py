import os
for experiment in [
    'density',
    'duration_shrink',
    'embedding_interpolation',
    'time_scale',
    'time_shift'
]:
    command = f'python final_tests/{experiment}.py'
    print(command)
    os.system(command)