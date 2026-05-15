from .config import DIC_AGENTS
import pickle
import os
import time
import traceback
import random


class Updater:

    def __init__(self, cnt_round, dic_agent_conf, dic_traffic_env_conf, dic_path):
        self.cnt_round = cnt_round
        self.dic_path = dic_path
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.agents = []
        self.sample_set_list = []
        self.sample_indexes = None

        if self.dic_traffic_env_conf["MODEL_NAME"] != "MRELight":
            raise ValueError("This release only supports MRELight.")

        print("Number of agents: ", dic_traffic_env_conf["NUM_AGENTS"])

        for i in range(dic_traffic_env_conf["NUM_AGENTS"]):
            agent = DIC_AGENTS["MRELight"](
                self.dic_agent_conf,
                self.dic_traffic_env_conf,
                self.dic_path,
                self.cnt_round,
                intersection_id=str(i)
            )
            self.agents.append(agent)

    def load_sample_with_forget(self, i):
        try:
            sample_file_path = os.path.join(
                self.dic_path["PATH_TO_WORK_DIRECTORY"],
                "train_round",
                "total_samples_inter_{0}.pkl".format(i)
            )

            with open(sample_file_path, "rb") as sample_file:
                cur_sample_set = []
                while True:
                    try:
                        cur_sample_set += pickle.load(sample_file)
                    except EOFError:
                        print("===== load samples finished =====")
                        break

            ind_end = len(cur_sample_set)
            ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
            memory_after_forget = cur_sample_set[ind_sta:ind_end]

            print("==== memory size after forget ====:", len(memory_after_forget))

            if self.cnt_round % self.dic_traffic_env_conf["FORGET_ROUND"] == 0:
                with open(sample_file_path, "wb+") as f:
                    pickle.dump(memory_after_forget, f, -1)

            sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))

            if self.sample_indexes is None:
                self.sample_indexes = random.sample(range(len(memory_after_forget)), sample_size)

            sample_set = [memory_after_forget[k] for k in self.sample_indexes]

            print("==== memory samples number =====:", sample_size)

        except Exception:
            error_dir = os.path.join(
                self.dic_path["PATH_TO_WORK_DIRECTORY"]
            ).replace("records", "errors")

            if not os.path.exists(error_dir):
                os.makedirs(error_dir)

            with open(os.path.join(error_dir, "error_info_inter_{0}.txt".format(i)), "a") as f:
                f.write("Fail to load samples for inter {0}\n".format(i))
                f.write("traceback.format_exc():\n%s\n" % traceback.format_exc())

            print("traceback.format_exc():\n%s" % traceback.format_exc())
            raise

        if i % 100 == 0:
            print("load_sample for inter {0}".format(i))

        return sample_set

    def load_sample_for_agents(self):
        start_time = time.time()
        print("Start load samples at", start_time)

        # MRELight uses samples from all intersections.
        samples_list = []
        for i in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            sample_set = self.load_sample_with_forget(i)
            samples_list.append(sample_set)

        self.agents[0].prepare_Xs_Y(samples_list)

    def update_network(self, i):
        print("update agent %d" % i)
        self.agents[i].train_network()
        self.agents[i].save_network(
            "round_{0}_inter_{1}".format(
                self.cnt_round,
                self.agents[i].intersection_id
            )
        )

    def update_network_for_agents(self):
        print("update_network_for_agents", self.dic_traffic_env_conf["NUM_AGENTS"])

        for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):
            self.update_network(i)