import subprocess
import sys

file_path = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"
bert_name = "uncased_L-12_H-768_A-12.zip"
result = subprocess.run(["wget", file_path])
result = subprocess.run(["unzip", bert_name])
result = subprocess.run(["rm", bert_name])
result = subprocess.run(["mkdir", "init_bert"])
result = subprocess.run(["mv", "uncased_L-12_H-768_A-12", "init_bert/"])


