from .config import DIC_AGENTS
from copy import deepcopy
from .cityflow_env import CityFlowEnv
import json
import os
import traceback


def test(model_dir, cnt_round, run_cnt, _dic_traffic_env_conf):
    dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)

    if dic_traffic_env_conf["MODEL_NAME"] != "MRELight":
        raise ValueError("This release only supports MRELight.")

    if dic_traffic_env_conf["NUM_AGENTS"] != 1:
        raise ValueError("MRELight release expects NUM_AGENTS = 1.")

    records_dir = model_dir.replace("model", "records")
    model_round = "round_%d" % cnt_round

    dic_path = {
        "PATH_TO_MODEL": model_dir,
        "PATH_TO_WORK_DIRECTORY": records_dir
    }

    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)

    anon_env_conf_path = os.path.join(records_dir, "anon_env.conf")

    if os.path.exists(anon_env_conf_path):
        with open(anon_env_conf_path, "r") as f:
            dic_traffic_env_conf = json.load(f)

        dic_traffic_env_conf["RUN_COUNTS"] = run_cnt

    if dic_traffic_env_conf["MODEL_NAME"] != "MRELight":
        raise ValueError("This release only supports MRELight.")

    if dic_traffic_env_conf["NUM_AGENTS"] != 1:
        raise ValueError("MRELight release expects NUM_AGENTS = 1.")

    dic_agent_conf["EPSILON"] = 0
    dic_agent_conf["MIN_EPSILON"] = 0

    agent = DIC_AGENTS["MRELight"](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path,
        cnt_round=0,
        intersection_id="0"
    )

    try:
        agent.load_network("{0}_inter_{1}".format(model_round, agent.intersection_id))

        path_to_log = os.path.join(
            dic_path["PATH_TO_WORK_DIRECTORY"],
            "test_round",
            model_round
        )

        if not os.path.exists(path_to_log):
            os.makedirs(path_to_log)

        env = CityFlowEnv(
            path_to_log=path_to_log,
            path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=dic_traffic_env_conf
        )

        done = False
        step_num = 0
        total_time = dic_traffic_env_conf["RUN_COUNTS"]
        state = env.reset()

        while not done and step_num < int(total_time / dic_traffic_env_conf["MIN_ACTION_TIME"]):
            # MRELight uses the global state of all intersections.
            action_list = agent.choose_action(step_num, state)

            next_state, reward, done, _ = env.step(action_list)

            state = next_state
            step_num += 1

        env.batch_log_2()
        env.end_cityflow()

    except Exception:
        print("============== error occurs in model_test ============")
        print(traceback.format_exc())
        raise