import pandas as pd
import glob
import json
import os
import time
from datetime import datetime

def time_string_to_seconds(timestamp):
    hh,mm,s = timestamp.split(':')
    ss,ms = s.split('.')
    time = 3600*int(hh) +  60*int(mm) + int(ss) + int(ms)/1000
    return time

def convert_clip_list(clip_list):
    return [[time_string_to_seconds(x) for x in clip] for clip in clip_list]

###### Change your path
parquet_dir = "/path/to/my/metadata/dir/" 

data = []
for jsonl in sorted(glob.glob(f"{parquet_dir}*.jsonl")):
    path = os.path.join(parquet_dir, jsonl)
    with open(path, "r") as f:
        for line in f:
            json_obj = json.loads(line)
            clips = [
                json_obj['clip'][i]['span']
                for i in range(len(json_obj['clip']))
            ]

            out = {
                'video_id': json_obj['video_id'],
                'url': json_obj['url'],
                'clips': clips
            }
            data.append(out)

df = pd.DataFrame(data)
df['clips'] = df['clips'].map(lambda x: convert_clip_list(x))
df.to_parquet("hd_vila.parquet")