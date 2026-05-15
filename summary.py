import argparse
import json
import os

import numpy as np
import pandas as pd


def get_metrics(duration_list, traffic_name, total_summary_metrics, num_of_out, validation_rounds=10):
    """
    Calculate summary metrics over the last validation_rounds test rounds.
    """
    if len(duration_list) == 0:
        return total_summary_metrics

    duration_array = np.array(duration_list, dtype=float)
    throughput_array = np.array(num_of_out, dtype=float)

    validation_duration = duration_array[-validation_rounds:]
    validation_throughput = throughput_array[-validation_rounds:]

    valid_duration = validation_duration[validation_duration > 0]

    final_duration = np.round(np.mean(valid_duration), decimals=2) if len(valid_duration) > 0 else np.nan
    final_duration_std = np.round(np.std(valid_duration), decimals=2) if len(valid_duration) > 0 else np.nan
    final_throughput = np.round(np.mean(validation_throughput), decimals=2) if len(validation_throughput) > 0 else np.nan

    total_summary_metrics["traffic"].append(traffic_name.replace(".json", ""))
    total_summary_metrics["final_duration"].append(final_duration)
    total_summary_metrics["final_duration_std"].append(final_duration_std)
    total_summary_metrics["final_through"].append(final_throughput)

    return total_summary_metrics


def read_vehicle_file(vehicle_file, run_counts):
    """
    Read one vehicle_inter_*.csv file and compute vehicle duration.
    """
    df_vehicle = pd.read_csv(
        vehicle_file,
        sep=",",
        header=0,
        dtype={0: str, 1: float, 2: float},
        names=["vehicle_id", "enter_time", "leave_time"]
    )

    df_vehicle["vehicle_id"] = df_vehicle["vehicle_id"].astype(str)

    # Remove CityFlow shadow vehicles.
    df_vehicle = df_vehicle[~df_vehicle["vehicle_id"].str.contains("shadow", na=False)].copy()

    # Keep the original leave_time to count completed vehicles.
    df_vehicle["leave_time_origin"] = df_vehicle["leave_time"]

    # Vehicles that have not left the network are treated as leaving at run_counts.
    df_vehicle["leave_time"] = df_vehicle["leave_time"].fillna(run_counts)
    df_vehicle["duration"] = df_vehicle["leave_time"] - df_vehicle["enter_time"]

    return df_vehicle


def summarize_one_round(round_dir, num_intersections, run_counts):
    """
    Summarize one test round over all intersections.
    """
    df_vehicle_all = []

    for inter_index in range(num_intersections):
        vehicle_file = os.path.join(round_dir, "vehicle_inter_{0}.csv".format(inter_index))

        if not os.path.exists(vehicle_file):
            print("Missing file:", vehicle_file)
            continue

        df_vehicle_inter = read_vehicle_file(vehicle_file, run_counts)

        if len(df_vehicle_inter) == 0:
            continue

        ave_duration_inter = df_vehicle_inter["duration"].mean(skipna=True)
        print("------------- inter_index: {0}\tave_duration: {1}".format(
            inter_index, ave_duration_inter
        ))

        df_vehicle_all.append(df_vehicle_inter)

    if len(df_vehicle_all) == 0:
        return None, 0, 0

    df_vehicle_all = pd.concat(df_vehicle_all, axis=0)

    # A vehicle may pass through multiple intersections, so aggregate its total travel time.
    vehicle_duration = df_vehicle_all.groupby(by=["vehicle_id"])["duration"].sum()
    ave_duration = vehicle_duration.mean()

    num_vehicle_in = len(df_vehicle_all["vehicle_id"].unique())
    num_vehicle_out = len(
        df_vehicle_all[df_vehicle_all["leave_time_origin"].notna()]["vehicle_id"].unique()
    )

    return ave_duration, num_vehicle_in, num_vehicle_out


def summary_detail_mrelight(memo, validation_rounds=10, records_root="records", summary_root="summary"):
    """
    Summarize MRELight test results.

    Expected record structure:
    records/
    └── <memo>/
        └── MRELight_<traffic_file>_<timestamp>/
            ├── traffic_env.conf
            └── test_round/
                ├── round_0/
                │   ├── vehicle_inter_0.csv
                │   └── ...
                └── round_1/
                    ├── vehicle_inter_0.csv
                    └── ...
    """
    total_summary = {
        "traffic": [],
        "final_duration": [],
        "final_duration_std": [],
        "final_through": [],
    }

    records_dir = os.path.join(records_root, memo)

    if not os.path.exists(records_dir):
        raise FileNotFoundError("Records directory does not exist: {}".format(records_dir))

    for experiment_dir in sorted(os.listdir(records_dir)):
        experiment_path = os.path.join(records_dir, experiment_dir)

        if not os.path.isdir(experiment_path):
            continue

        traffic_env_conf_path = os.path.join(experiment_path, "traffic_env.conf")
        test_round_dir = os.path.join(experiment_path, "test_round")

        if not os.path.exists(traffic_env_conf_path):
            continue

        if not os.path.exists(test_round_dir):
            print("No test_round directory in {}".format(experiment_dir))
            continue

        with open(traffic_env_conf_path, "r") as f:
            dic_traffic_env_conf = json.load(f)

        model_name = dic_traffic_env_conf.get("MODEL_NAME", "MRELight")
        if model_name != "MRELight":
            print("Skip non-MRELight experiment:", experiment_dir)
            continue

        run_counts = dic_traffic_env_conf["RUN_COUNTS"]
        num_intersections = dic_traffic_env_conf["NUM_INTERSECTIONS"]
        traffic_name = dic_traffic_env_conf.get("TRAFFIC_FILE", experiment_dir)

        print("========================================")
        print("Experiment:", experiment_dir)
        print("Traffic:", traffic_name)

        round_files = [
            f for f in os.listdir(test_round_dir)
            if f.startswith("round_") and os.path.isdir(os.path.join(test_round_dir, f))
        ]

        round_files.sort(key=lambda x: int(x.split("_")[1]))

        duration_each_round_list = []
        num_of_vehicle_in = []
        num_of_vehicle_out = []

        for round_name in round_files:
            round_dir = os.path.join(test_round_dir, round_name)

            ave_duration, vehicle_in, vehicle_out = summarize_one_round(
                round_dir=round_dir,
                num_intersections=num_intersections,
                run_counts=run_counts
            )

            if ave_duration is None:
                print("Empty round:", round_name)
                continue

            duration_each_round_list.append(ave_duration)
            num_of_vehicle_in.append(vehicle_in)
            num_of_vehicle_out.append(vehicle_out)

            print(
                "==== round: {0}\tave_duration: {1}\tnum_vehicle_in: {2}\tnum_vehicle_out: {3}".format(
                    round_name,
                    ave_duration,
                    vehicle_in,
                    vehicle_out
                )
            )

        if len(duration_each_round_list) == 0:
            print("No valid test results in {}".format(experiment_dir))
            continue

        result_dir = os.path.join(summary_root, memo, experiment_dir)
        os.makedirs(result_dir, exist_ok=True)

        round_result = pd.DataFrame({
            "duration": duration_each_round_list,
            "vehicle_in": num_of_vehicle_in,
            "vehicle_out": num_of_vehicle_out
        })

        round_result.to_csv(
            os.path.join(result_dir, "test_results.csv"),
            index=False
        )

        total_summary = get_metrics(
            duration_list=duration_each_round_list,
            traffic_name=traffic_name,
            total_summary_metrics=total_summary,
            num_of_out=num_of_vehicle_out,
            validation_rounds=validation_rounds
        )

        total_result = pd.DataFrame(total_summary)
        total_result.to_csv(
            os.path.join(summary_root, memo, "total_test_results.csv"),
            index=False
        )

    return pd.DataFrame(total_summary)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str, default="benchmark_1001")
    parser.add_argument("--validation_rounds", type=int, default=10)
    parser.add_argument("--records_root", type=str, default="records")
    parser.add_argument("--summary_root", type=str, default="summary")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary_detail_mrelight(
        memo=args.memo,
        validation_rounds=args.validation_rounds,
        records_root=args.records_root,
        summary_root=args.summary_root
    )