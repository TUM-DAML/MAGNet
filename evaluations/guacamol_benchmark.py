import argparse

from pymongo import MongoClient

from models.eval_utils import *
from models.global_utils import SEML_CONFIG_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="baselines-sample")
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
            if isinstance(r["result"], dict):
                continue
            if len(r["result"]) > 50000:
                print("too many smiles")
                continue
            print(r["config"]["model_name"])
            smiles = r["result"]
            result = calculate_all_sampling_metrics(smiles, "zinc")
            result["generated_smiles"] = smiles
            collection.update_one({"_id": r["_id"]}, {"$set": {"result": result}})
