# -*- coding: utf-8 -*-
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
os.environ["PATH"] = 'C:\\ProgramData\\Anaconda3;C:\\ProgramData\\Anaconda3\\Library\\mingw-w64\\bin;C:\\ProgramData\\Anaconda3\\Library\\usr\\bin;C:\\ProgramData\\Anaconda3\\Library\\bin;C:\\ProgramData\\Anaconda3\\Scripts;C:\\ProgramData\\Anaconda3\\bin;C:\\ProgramData\\Anaconda3\\condabin;C:\\ProgramData\\Anaconda3;C:\\ProgramData\\Anaconda3\\Library\\mingw-w64\\bin;C:\\ProgramData\\Anaconda3\\Library\\usr\\bin;C:\\ProgramData\\Anaconda3\\Library\\bin;C:\\ProgramData\\Anaconda3\\Scripts;C:\\Program Files (x86)\\Common Files\\Oracle\\Java\\javapath;C:\\ProgramData\\Oracle\\Java\\javapath;C:\\Program Files (x86)\\Common Files\\Intel\\Shared Libraries\\redist\\intel64\\compiler;C:\\Program Files (x86)\\Intel\\iCLS Client;C:\\Program Files\\Intel\\iCLS Client;C:\\Windows\\system32;C:\\Windows;C:\\Windows\\System32\\Wbem;C:\\Windows\\System32\\WindowsPowerShell\\v1.0;C:\\Program Files (x86)\\NVIDIA Corporation\\PhysX\\Common;C:\\Program Files (x86)\\Intel\\Intel(R) Management Engine Components\\DAL;C:\\Program Files\\Intel\\Intel(R) Management Engine Components\\DAL;C:\\Program Files (x86)\\Intel\\Intel(R) Management Engine Components\\IPT;C:\\Program Files\\Intel\\Intel(R) Management Engine Components\\IPT;C:\\Program Files\\Microsoft SQL Server\\110\\Tools\\Binn;C:\\WINDOWS\\system32;C:\\WINDOWS;C:\\WINDOWS\\System32\\Wbem;C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0;C:\\Users\\S4\\.dnx\\bin;C:\\Program Files\\Microsoft DNX\\Dnvm;C:\\Program Files (x86)\\Windows Kits\\8.1\\Windows Performance Toolkit;D:\\Program Files\\Git\\cmd;C:\\Program Files\\CMake\\bin;C:\\Program Files\\Common Files\\Autodesk Shared;C:\\Program Files (x86)\\Autodesk\\Backburner;C:\\Program Files (x86)\\Graphviz2.38\\bin;D:\\3rd_party_library\\ffmpeg-20171219-c94b094-win64-static\\bin;D:\\3rd_party_library\\wechat_callback-master;C:\\Program Files\\NVIDIA Corporation\\NVSMI;C:\\WINDOWS\\System32\\OpenSSH;C:\\Program Files\\SlikSvn\\bin;C:\\Program Files\\MiKTeX 2.9\\miktex\\bin\\x64;C:\\Program Files\\IDM Computer Solutions\\UltraCompare;C:\\Program Files\\PuTTY;C:\\Program Files\\NVIDIA Corporation\\NVIDIA NvDLISR;PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\include;C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin;C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\libx64;C:\\tools\\cuda\\bin;.;C:\\WINDOWS\\system32;C:\\WINDOWS;C:\\WINDOWS\\System32\\Wbem;C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0;C:\\WINDOWS\\System32\\OpenSSH;C:\\Program Files (x86)\\QuickTime\\QTSystem;C:\\Strawberry\\c\\bin;C:\\Strawberry\\perl\\site\\bin;C:\\Strawberry\\perl\\bin;C:\\Program Files\\dotnet;C:\\Program Files (x86)\\pcsuite;C:\\texlive\\2022\\bin\\win32;D:\\texlive\\2019\\bin\\win32;D:\\tools\\python3_5;C:\\Users\\S4\\AppData\\Local\\Microsoft\\WindowsApps;D:\\3rd_party_library\\ffmpeg-20171219-c94b094-win64-static\\bin;D:\\3rd_party_library\\wechat_callback-master;C:\\Program Files\\NVIDIA Corporation\\NVSMI;D:\\Work_Space\\Project_2018\\boltzmann-machines-master\\bm;D:\\Work_Space\\Project_2018\\boltzmann-machines-master\\bm\\base;D:\\Work_Space\\Project_2018\\boltzmann-machines-master\\bm\\rbm;D:\\Work_Space\\Project_2018\\boltzmann-machines-master\\bm\\utils;D:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\bin;C:\\texlive\\2018\\bin\\win32;D:\\Qt\\Qt5.7.1\\5.7\\mingw53_32\\bin;C:\\Users\\S4\\AppData\\Local\\GitHubDesktop\\bin;C:\\Users\\S4\\AppData\\Local\\atom\\bin;.;C:\\Users\\S4\\AppData\\Local\\Microsoft\\WindowsApps;D:\\Microsoft VS Code\\bin'
import faiss
import numpy as np
from sklearn.preprocessing import normalize


def get_all_answers(model,dataset,question_tpye,infor):
	answer_list = "../output_data/%s/answers_%s_%s_%s/"%(model,dataset,question_tpye,"infor" if infor else "noninfor")
	file_list = os.listdir(answer_list)
	txt = ""
	for file in file_list:
		with open(answer_list+file, "r") as f:  # 打开文件
			   text = f.read()
		txt = txt+text
	ttxtt = txt.split("\n")
	tt = []
	for t in ttxtt:
		if len(t)>40:
			tx = t.split(".")
			for txx in tx:
				if len(txx)>30:
					tt.append(txx.strip( ' ' ))
			tt.pop(-1)
	return tt

def cal_mean_score(questions,key,f):
	#questions, the dataframe
	#key: name of the facet or factor
	#f = 'factor', or 'facet'
	if f == "facet":
		key = key.title()
	score = np.mean(questions[questions[f]==key]['score'])
	return score

def main():
	dataset = "opva"
	question_tpyes = ["factors","hirability"]
	infor = True
	model = "gpt-4"
	model = SentenceTransformer('all-MiniLM-L6-v2')
	questions = pd.read_excel("HEXACO-60.xlsx")
	items =[]
	dic = 	{"H":"honesty-humility","E":"emotionality","X":"extraversion","A":"agreeableness","C":"conscientiousness","O":"openness to experience"}
	dic2= {"sinc":"Sincerity",
	"fair":"Fairness",
	"gree":"Greed Avoidance",
	"mode":"Modesty",
	"fear":"Fearfulness",
	"anxi":"Anxiety",
	"depe":"Dependence",
	"sent":"Sentimentality",
	"sses":"Social Self-Esteem",
	"socb":"Social Boldness",
	"soci":"Sociability",
	"live":"Liveliness",
	"forg":"Forgivingness",
	"gent":"Gentleness",
	"flex":"Flexibility",
	"pati":"Patience",
	"orga":"Organization",
	"dili":"Diligence",
	"perf":"Perfectionism",
	"prud":"Prudence",
	"aesa":"Aesthetic Appreciation",
	"inqu":"Inquisitiveness",
	"crea":"Creativity",
	"unco":"Unconventionality"}
	
	for question_type in question_tpyes:
		factors = []
		facets  = []
		tt = get_all_answers(model,dataset,question_type,infor)
		for index, q in questions.iterrows():
			question = q["Questions"]
			r = "low" if q["Rec"] == "R" else "high"
			f = q["Name"][0]
			f2 = q["Name"][1:5]
			factors.append(dic[f])
			facets.append(dic2[f2])
			head = "This individual has %s %s if "%(r,dic[f])
			items.append(head+question)	
		embeddings = model.encode(items)
		embeddings_e = model.encode(tt)
		d = 384
		index = faiss.IndexFlatL2(d)
		index.add(embeddings)
		k = 60
		D, I = index.search(embeddings_e, k) 
		index = np.zeros([k,1])
		data = np.subtract(1,normalize(D, axis=1, norm='max'))
		for d in range(I.shape[0]):
			index[I[d,0],0] = index[I[d,0],0]+data[I[d,0],0]
		index_ = np.zeros([k,1])
		index_number = np.zeros([k,1])
		score_name = 'score_'+question_type
		index__ = np.divide(index_,index_number)
		index__ = np.nan_to_num(index__)
		questions[score_name]=index__
		print("========similarity scores for %s=============="%question_type)
		for factor in list(dic.values()):
			score = np.mean(questions[questions['factor']==factor][score_name])
			print(factor+": %.2f"%score)
		print("========================================================")
		for facet in list(dic2.values()):
			score = np.mean(questions[questions['facet']==facet.lower()][score_name])
			print(facet+":%.2f"%score)

if __name__ == "__main__": 
	main()
	os.system('pause')
