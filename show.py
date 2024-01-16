import pandas as pd
import os

# 定义包含多个CSV文件路径的列表
csv_paths = [
    "/data1/zhangjiaming/results_Tracking_Anything/v4_convmae_mask2former/1230-v3.yaml/Jan08_16.07.13_v4_convmae_mask2former_1230-v3_s6/Jan08_16.07.13_v4_convmae_mask2former_1230-v3_s6_450000.pth/",
]

result_name = 'TrackingAnything_1230-test.txt-memory_store_every{5}-memory_max_number{1}-model_topk{30}'
csv_name = 'results.csv'

for path in csv_paths:
    csv_path = os.path.join(path, result_name, csv_name)
    try:
        df = pd.read_csv(csv_path)
        print(df.iloc[0, 2])

    except Exception as e:
        print(f"Error reading {csv_path}", )
