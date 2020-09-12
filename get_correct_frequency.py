import pandas as pd
import numpy as np
import sys
import os

def main_frequencies(folder_name, model_name):
    fnames = [
        f
        for f in os.listdir(folder_name)
        if "_" + model_name + ".csv" in f and f.endswith(".csv")
    ]
    paths = [os.path.join(folder_name, f) for f in fnames]
    diffnum_files = [
         f
         for f in paths
         if "diffnum_direct" in f
         if os.path.exists(f.replace("direct", "indirect"))
    ]

    frequency_ratios = []
    frequency_ratios_diffnum = []
    for path in diffnum_files:
        temp_df = pd.read_csv(path).groupby("base_string").agg("mean").reset_index()
        temp_df["correct"] = temp_df["candidate1_base_prob"] < temp_df["candidate2_base_prob"]
        temp_df["diffnum_correct"] = temp_df["candidate1_alt1_prob"] > temp_df["candidate2_alt1_prob"]
        print(temp_df["correct"], temp_df["diffnum_correct"])
        num_correct = temp_df.correct.sum()
        num_incorrect = len(temp_df) - num_correct
        num_correct_diffnum = temp_df.diffnum_correct.sum()
        num_incorrect_diffnum = len(temp_df) - num_correct_diffnum
        print(path, num_correct / (num_correct + num_incorrect), num_correct_diffnum / (num_correct_diffnum + num_incorrect_diffnum))
        frequency_ratios.append(num_correct / (num_correct + num_incorrect))
        frequency_ratios_diffnum.append(num_correct_diffnum / (num_correct_diffnum + num_incorrect_diffnum))

    print("Overall accuracy for", model_name, ":", np.mean(frequency_ratios), " | ", np.mean(frequency_ratios_diffnum))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: python ", sys.argv[0], "<folder_name> <model_name>")
    folder_name = sys.argv[1]
    model_name = sys.argv[2]

    main_frequencies(folder_name, model_name)
