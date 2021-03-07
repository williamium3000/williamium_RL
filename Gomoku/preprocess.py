import numpy as np


for path in ["data_set/gomocup/dataset_gomocup15_tfrec_a_one/train_a_one.npz", "data_set/gomocup/dataset_gomocup15_tfrec_a_one/test_a_one.npz", "data_set/gomocup/dataset_gomocup15_tfrec_a_one/validation_a_one.npz"]:
    data = np.load(path)


    states = data["states"]
    actions = data["actions"]

    for i in range(states.shape[0]):
        if i % 1000:
            print(i)

        state = states[i]
        
        if state[state != 0].sum() % 2 == 0:
            current_player = 1
        else:
            current_player = 2
        if current_player == 2:
            state[state == 2] = 3
            state[state == 1] = 2
            state[state == 3] = 1
        states[i] = state
    np.savez(path.split(".")[0] + "_preprocessed.npz", states = states, actions = actions)
