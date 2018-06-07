import os
import pickle
import numpy as np
import pandas as pd

class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))

def dir_to_df(dir_path = "", visual_index = []):
    i = 0
    df = {}
    for file_name in os.listdir(dir_path):
        file_path = dir_path + "/" +file_name
        file_content = np.load(file_path)
        file_content = np.reshape(file_content, -1)
        file_name_content = {visual_index[0]:file_name, visual_index[1]:file_content }
        df[i] = file_name_content
        i += 1
    df = pd.DataFrame.from_dict(df, orient="index")
    return df

def file_to_df(file_path = "", file_index = []):
    with open(file_path, 'r') as file:
        i = 0
        df = {}
        for line in file:
            j = 0
            line_dic = {}
            line = line.rstrip("\n")
            items = line.split("\t")
            for item in items:
                line_dic[file_index[j]] = item
                j += 1
            df[i] = line_dic
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df

def face_file_to_df(face_file_path = "", file_index = []):
    with open(face_file_path, 'r') as file:
        i = 0
        df ={}
        for line in file:
            j = 0
            line_dic = {}
            line = line.rstrip("\n")
            items = line.split("\t")
            items = list(items)
            for item in items:
                if j == 0:
                    line_dic[file_index[j]] = item
                    j += 1
                else:
                    item = eval(item)
                    for k in range(len(item[0])):
                        line_dic[file_index[j]] = item[0][k]
                        j += 1
            df[i] = line_dic
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df

