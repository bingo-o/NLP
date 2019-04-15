# 需要从百度网盘：https://pan.baidu.com/s/1siRK4uZn9SwzDQOmAY2GLg 提取码：i5mt 
# 下载压缩包解压到StanfordCoreNLP文件夹下即可

from stanfordcorenlp import StanfordCoreNLP
import json
import os
from tqdm import tqdm

def text_processing(text):
    punct = '''`~!@#$%&*()-_+={}|[]\\:;'"<>,.?/—,'''
    for sign in punct:
        text = text.replace(sign, " " + sign + " ")
    return text

input_dir = "data_input"
output_dir = "data_output"
input_files = os.listdir(input_dir)
with StanfordCoreNLP("../../BigFile/stanford-corenlp-full-2018-10-05") as nlp:
    for file in tqdm(input_files):
        with open(input_dir + "/" + file, "r", encoding="utf-8") as f:
            text = f.readline().strip()
        text = text_processing(text)
        words = nlp.word_tokenize(text)
        text = " ".join(words)
        props = {'annotators': 'tokenize,pos,lemma,depparse', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
        info = json.loads(nlp.annotate(text, properties=props))
        with open(output_dir + "/" + file, "w", encoding="utf-8") as f:
            for i in range(len(info["sentences"])):
                tokens = info["sentences"][i]["tokens"]
                deps = sorted(info["sentences"][i]["basicDependencies"], key=lambda x: x["dependent"])
                for token, dep in zip(tokens, deps):
                    line = token["word"] + "\t" + str(token["characterOffsetBegin"]) + "\t" + str(
                        token["characterOffsetEnd"]) + "\t" + token["pos"] + "\t" + token["lemma"] + "\t" + str(
                        dep["governor"]) + "\t" + dep["dep"] + "\n"
                    f.write(line)
