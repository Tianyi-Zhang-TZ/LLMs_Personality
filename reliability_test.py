# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import os
from sklearn.metrics import r2_score, accuracy_score,mean_absolute_error
import warnings
from scipy.stats import pearsonr
import data_analysis
warnings.filterwarnings("ignore")

def compare_metric_repeat_measures(ground_truth,predcitions,select_col,c_metric):
	if ground_truth.index[0] == 0:	
		ground_truth.index = ground_truth["participantid"]
	if predcitions.index[0] == 0:	
		predcitions.index = predcitions["participantid"]
	select_col_tru = select_col_pre		
	for i in range(len(select_col)):		
		ground_truth.loc[:,select_col_tru[i]] = pd.to_numeric(ground_truth[select_col_tru[i]],errors='coerce')
		predcitions.loc[:,select_col_pre[i]] = pd.to_numeric(predcitions[select_col_pre[i]],errors='coerce')
	ground_truth = ground_truth.dropna()
	predcitions = predcitions.dropna()
	common_ids = ground_truth.index.intersection(predcitions.index)	
	tru_data = ground_truth.loc[common_ids,select_col]
	pre_data = predcitions.loc[common_ids,select_col]
	for i in range(len(select_col_pre)):
		if c_metric =='r2':
			m = r2_score(tru_data[select_col_tru[i]], pre_data[select_col_pre[i]])
			print("%s for %s = %s"%(c_metric,select_col_pre[i],round(m,2)))
		elif  c_metric =='accuracy':
			m = accuracy_score(tru_data[select_col_tru[i]].round(0), pre_data[select_col_pre[i]].round(0))
			print("%s for %s = %s%%"%(c_metric,select_col_pre[i],round(m*100,2)))
		elif c_metric =="MAE":
			m = 1-mean_absolute_error(tru_data[select_col_tru[i]], pre_data[select_col_pre[i]])
			print("%s for %s = %s"%(c_metric,select_col_pre[i],round(m,2)))
		elif c_metric == "corr":
			m,n = pearsonr(pre_data[select_col_pre[i]],tru_data[select_col_tru[i]])
			print("------------------------------------")
			print("%s for %s = %s"%(c_metric,select_col_pre[i],round(m,2)))
			print("p-value for %s = %s"%(select_col_pre[i],round(n,2)))
	return tru_data,pre_data,m	

def get_data_repeat_measures(model,dataset,question_tpyes,n,infor,t):#get the grond truth and predictions 
	#t="",first round of test, t = "2" second round of test
	question_tpye = question_tpyes[n] if n !=4 else "facets"
	answer_list = "output_data/%s/answers_%s_%s_%s%s/"%(model,dataset,question_tpye,"infor" if infor else "noninfor",t)
	if n !=4:
		pre_dir = "output_data/%s/pre_%s_%s_%s%s.pkl"%(model,dataset,question_tpye,"infor" if infor else "noninfor",t)
		if os.path.exists(pre_dir):
			f = open(pre_dir,"rb")
			predicts = pickle.load(f)
			f.close()
		else:
			predicts,pre_dir = data_analysis.get_from_answers(dataset,question_tpye,model,infor,answer_list)		
	else:
		#for mean of facets only
		pre_dir = "output_data/%s/pre_%s_%s_%s%s.pkl"%(model,dataset,"mean_facets","infor" if infor else "noninfor",t)
		if os.path.exists(pre_dir):
			f = open(pre_dir,"rb")
			predicts = pickle.load(f)
			f.close()
		else:
			predictions,_ = data_analysis.get_from_answers(dataset,question_tpye,model,infor,answer_list)	
			predicts = data_analysis.mean_facets(predictions,pre_dir)
	f = open("data_files/data.pkl",'rb')
	data = pickle.load(f)
	f.close()
	return predicts, data,pre_dir

if __name__ == "__main__": 
	metric = "corr"
	dataset = "opva"
	question_tpyes = ["factors","facets","factors_all","hirability","mean_facets"]
	infor = True
	models = ["gpt-3.5","gpt-4"]	 
	for model in models:
		for n in [0,3]:
			question_tpye = question_tpyes[n] if n !=4 else "facets"
			m = 0 if n>=4 else n	
			predicts,data,pre_dir = get_data_repeat_measures(model,dataset,question_tpyes,n,infor,"")
			ground_truth,_,_ = get_data_repeat_measures(model,dataset,question_tpyes,n,infor,"2")#for repeat measures
			test_retest,_,_ = get_data_repeat_measures(model,dataset,question_tpyes,n,infor,"_testretest") #for test_retest
			sc_pre = [["Extraversion","Conscientiousness"],
					  ['Social self-esteem','Social boldness','Sociability','Liveliness','Organization','Diligence','Prudence','Perfectionism'],
					  ["Honesty-Humility","Emotionality","Extraversion","Agreeableness","Conscientiousness","Openness to Experience"],
					  ['Development orientation','Communication flexibility','Persuasiveness','Quality orientation','Overall hireability']]
			select_col_pre =sc_pre [m]
			select_col_tru = select_col_pre# for testretest
			print ("====================================")
			print("\tmodel = %s\t\n\tquestion_tpye = %s \t"%(model,question_tpyes[n]))
			print ("====================================")
			print("\tRepeated measures\t")
			tru_data,pre_data,w = compare_metric_repeat_measures(ground_truth,predicts,select_col_pre,"corr")
			print("------------------------------------")
			print("\tTest-retest\t")
			tru_data,pre_data,w = compare_metric_repeat_measures(test_retest,predicts,select_col_pre,"corr")
			print("------------------------------------",end="\n\n")
	os.system('pause')
