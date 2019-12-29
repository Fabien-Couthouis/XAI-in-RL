# TODO

- Load trained pytorch model (particles env) and make actions predictions 
- Code for agent contribution calculation (particles env for simple_tag task): 
    - Calculate all possible coalitions 
    - For each coalition: launch simulations (with absent agents play at random) and collect mean reward
    - Calculate Shapley values (approximations not necessary for this env)
    - Plot it meaninfully (?)
- Update env.yml & install instructions
