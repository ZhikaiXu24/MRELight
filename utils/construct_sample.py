import numpy as np
import pickle
import os
import traceback

def get_reward_from_features(rs):
    # 将12个道路的排队长度加在一起
    reward = {
        "queue_length": np.sum(rs["lane_num_waiting_vehicle_in"]),
        "average_speed": round(np.mean(rs["average_vehicle_speed"])),
        "pressure": np.sum(np.abs(rs["pressure"]))
    }
    return reward

def cal_reward(rs, rewards_components):
    r = 0
    for component, weight in rewards_components.items():
        if weight == 0:
            continue
        if component not in rs.keys():
            continue
        if rs[component] is None:
            continue
        r += rs[component] * weight
    return r


class ConstructSample:

    def __init__(self, path_to_samples, cnt_round, dic_traffic_env_conf):
        self.parent_dir = path_to_samples
        self.path_to_samples = path_to_samples + "/round_" + str(cnt_round)
        self.cnt_round = cnt_round
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.logging_data_list_per_gen = None
        self.hidden_states_list = None
        self.samples = []
        self.samples_all_intersection = [None]*self.dic_traffic_env_conf['NUM_INTERSECTIONS']

        self.interval = self.dic_traffic_env_conf["MIN_ACTION_TIME"]
        self.measure_time = self.dic_traffic_env_conf["MEASURE_TIME"]

    def load_data(self, folder, i):
        try:
            f_logging_data = open(os.path.join(self.path_to_samples, folder, "inter_{0}.pkl".format(i)), "rb")
            logging_data = pickle.load(f_logging_data)
            f_logging_data.close()
            return 1, logging_data

        except Exception:
            print("Error occurs when making samples for inter {0}".format(i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0, None

    def load_data_for_system(self, folder):
        self.logging_data_list_per_gen = []
        print("Load data for system in ", folder)
        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            pass_code, logging_data = self.load_data(folder, i)
            if pass_code == 0:
                return 0
            self.logging_data_list_per_gen.append(logging_data)
        return 1

    def construct_state(self, features, time, i):
        state = self.logging_data_list_per_gen[i][time]
        assert time == state["time"]
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            state_after_selection = {}
            for key, value in state["state"].items():
                if key in features:
                    if "cur_phase" in key:
                        state_after_selection[key] = self.dic_traffic_env_conf['PHASE'][value[0]]
                    else:
                        state_after_selection[key] = value
        else:
            state_after_selection = {key: value for key, value in state["state"].items() if key in features}
        return state_after_selection

    def construct_reward(self, rewards_components, time, i):
        rs = self.logging_data_list_per_gen[i][time + self.measure_time - 1]
        assert time + self.measure_time - 1 == rs["time"]
        rs = get_reward_from_features(rs['state'])

        if "pressure" in rewards_components:
            rewards_components_1 = {"pressure": rewards_components.get("pressure", -0.25)}
            rewards_components_2 = {"queue_length": 0, "average_speed": 0}
        else:
            rewards_components_1 = {'queue_length': rewards_components.get('queue_length', -1), 'average_speed': 0}
            rewards_components_2 = {'queue_length': 0, 'average_speed': rewards_components.get('average_speed', 1)}

        r_instant_1 = cal_reward(rs, rewards_components_1)
        r_instant_2 = cal_reward(rs, rewards_components_2)
        # average15个time_step的平均奖励
        list_r_1 = []
        list_r_2 = []
        for t in range(time, time + self.measure_time):
            rs = self.logging_data_list_per_gen[i][t]
            assert t == rs["time"]
            rs = get_reward_from_features(rs['state'])
            r_1 = cal_reward(rs, rewards_components_1)
            r_2 = cal_reward(rs, rewards_components_2)
            list_r_1.append(r_1)
            list_r_2.append(r_2)
        r_average_1 = np.average(list_r_1)
        r_average_2 = np.average(list_r_2)  

        return r_instant_1,r_instant_2,r_average_1,r_average_2

    def judge_action(self, time, i):
        if self.logging_data_list_per_gen[i][time]['action'] == -1:
            raise ValueError
        else:
            return self.logging_data_list_per_gen[i][time]['action']

    def make_reward(self, folder, i):
        if self.samples_all_intersection[i] is None:
            self.samples_all_intersection[i] = []
        if i % 100 == 0:
            print("make reward for inter {0} in folder {1}".format(i, folder))
        list_samples = []
        try:
            total_time = int(self.logging_data_list_per_gen[i][-1]['time'] + 1)
            for time in range(0, total_time - self.measure_time + 1, self.interval):#interval=15
                state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"], time, i)
                reward_instant_1, reward_instant_2,reward_average_1,reward_average_2 = self.construct_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"],
                                                                       time, i)
                action = self.judge_action(time, i)
                #得出下一个状态
                if time + self.interval == total_time:
                    next_state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                                                      time + self.interval - 1, i)

                else:
                    next_state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                                                      time + self.interval, i)
                sample = [state, action, next_state, reward_average_1, reward_average_2,reward_instant_1,reward_instant_2, time,
                          folder+"-"+"round_{0}".format(self.cnt_round)]
                list_samples.append(sample)

            self.samples_all_intersection[i].extend(list_samples)
            return 1
        except:
            print("Error occurs when making rewards in generator {0} for intersection {1}".format(folder, i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0

    def make_reward_for_system(self):
        for folder in os.listdir(self.path_to_samples):
            print(folder)
            if "generator" not in folder:
                continue
            if not self.load_data_for_system(folder):
                continue
            for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                pass_code = self.make_reward(folder, i)
                if pass_code == 0:
                    continue

        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            self.dump_sample(self.samples_all_intersection[i], "inter_{0}".format(i))

    def dump_sample(self, samples, folder):
        if folder == "":
            with open(os.path.join(self.parent_dir, "total_samples.pkl"), "ab+") as f:
                pickle.dump(samples, f, -1)
        elif "inter" in folder:
            with open(os.path.join(self.parent_dir, "total_samples_{0}.pkl".format(folder)), "ab+") as f:
                pickle.dump(samples, f, -1)
        else:
            with open(os.path.join(self.path_to_samples, folder, "samples_{0}.pkl".format(folder)), 'wb') as f:
                pickle.dump(samples, f, -1)
