import argparse
import copy
import os

from pymongo import MongoClient

from baselines.eval_utils import *
from baselines.global_utils import SEML_CONFIG_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="baselines-zero-shot")
    args = parser.parse_args()

    access_dict = {}
    with open(SEML_CONFIG_PATH, "r") as f:
        for line in f.readlines():
            if len(line.strip()) > 0:
                split = line.split(":")
                key = split[0].strip()
                value = split[1].strip()
                access_dict[key] = value

    database = MongoClient(
        access_dict["host"],
        int(access_dict["port"]),
        username=access_dict["username"],
        password=access_dict["password"],
        authSource=access_dict["database"],
    )[access_dict["database"]]
    collection = database[args.collection]
    result = collection.find({})

    for r in result:
        if r["status"] == "COMPLETED":
            result_og = r["result"]
            if "tanimoto_scores" in result_og:
                continue
            output_smiles = result_og["reconstructed_smiles"]
            gt_smiles = result_og["gt_smiles"]
            # calculate tanimoto and exact match score and collect in list
            tanimoto_scores = []
            exact_match_scores = []
            for i in range(len(output_smiles)):
                if "." in output_smiles[i]:
                    continue
                tanimoto_scores.append(calculate_tanimoto(output_smiles[i], gt_smiles[i]))
                exact_match_scores.append(compare_smiles(output_smiles[i], gt_smiles[i]))
            print("original length: ", len(output_smiles))
            print("without disconnected: ", len(tanimoto_scores))
            result = dict()
            result["tanimoto_scores"] = tanimoto_scores
            result["exact_match_scores"] = exact_match_scores
            result["reconstructed_smiles"] = output_smiles
            result["gt_smiles"] = gt_smiles
            # delete tmp file
            # temporarily write ground truth files to file
            with open("tmp.out", "w") as file:
                for o in gt_smiles:
                    file.write(o + "\n")
            result.update(calculate_guacamol_benchmark("tmp.out", copy.deepcopy(output_smiles)))
            os.remove("tmp.out")
            collection.update_one({"_id": r["_id"]}, {"$set": {"result": result}})
