# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import numpy as np
import os
import data_analysis
import scipy.stats as stats
import pingouin as pg
import pyreadstat
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def get_compared_group(dataset,question_tpye,model,infor,answer_list,select_col_pre,select_col_tru,select_variables):
	pre_dir = "output_data/%s/pre_%s_%s_%s.pkl"%(model,dataset,question_tpye,"infor" if infor else "noninfor")
	if os.path.exists(pre_dir):
		f = open(pre_dir,"rb")
		predict = pickle.load(f)
		f.close()
	else:
		predict,file_name = data_analysis.get_from_answers(dataset,question_tpye,model,infor,answer_list)
	ground_truth = pd.read_csv('data_files/OSFdata_n710_20230220.csv')
	ground_truth.index = ground_truth["workerId"]
	select_col_tru = select_col_tru+select_variables
	ground_truth = ground_truth[select_col_tru]
	if predict.index[0] == 0:	
		predict.index = predict["participantid"]
	predcitions = predict[select_col_pre]
	for i in range(len(select_col_pre)):		
		ground_truth.loc[:,select_col_tru[i]] = pd.to_numeric(ground_truth[select_col_tru[i]],errors='coerce')
		predcitions.loc[:,select_col_pre[i]] = pd.to_numeric(predcitions[select_col_pre[i]],errors='coerce')
	ground_truth = ground_truth.dropna()
	predcitions = predcitions.dropna()
	common_ids = ground_truth.index.intersection(predcitions.index)	
	pre_data = predcitions.loc[common_ids,select_col_pre]
	tru_data = ground_truth.loc[common_ids,select_col_tru]
	tru_data.columns = ["tru_"+c for c in pre_data.columns]+select_variables
	df = pd.concat([pre_data,tru_data],axis=1)
	return df

def cohend(d1, d2):
 # calculate the size of samples
 n1, n2 = len(d1), len(d2)
 # calculate the variance of the samples
 s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
 # calculate the pooled standard deviation
 s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
 # calculate the means of the samples
 u1, u2 = np.mean(d1),np.mean(d2)
 # calculate the effect size
 return (u1 - u2) / s


def corr_df(df_norm,rating,variable):
	if variable =="gender":
		df_1 =  np.array(df_norm[df_norm["gender"]==1][rating])
		df_2 = np.array(df_norm[df_norm["gender"]==2][rating])
		d = cohend(df_1, df_2)
	else:
		d,_ = pearsonr(df_norm[rating],df_norm[variable])
	return d

	
def main():
	dataset = "opva"
	question_tpyes = ["factors","facets","facets_factors","factors_all"]
	question_tpye = question_tpyes[0]
	infor = True
	models = ["gpt-3.5","gpt-4"]
	ground_truth = pd.read_csv('data_files/OSFdata_n710_20230220.csv')
	d2 = ground_truth.columns
	sc = [[["AI_Extraversion_observer_facet_mean","AI_Conscientiousness_observer_facet_mean"],
			["Extraversion_observer_facet_mean","Conscientiousness_observer_facet_mean"],
				      ["extra10","consc10"]],
		   [d2[188:196],d2[118:126],d2[134:142], d2[150:158], d2[166:174],d2[444:452]],#facets
		   [d2[182:188],d2[112:118]]]
	
	select_col_tru = list(sc[0][1])
	select_col_pre = ["Extraversion","Conscientiousness"]
	#select_col_pre = ['Social self-esteem','Social boldness','Sociability','Liveliness','Organization','Diligence','Prudence','Perfectionism']	
	select_variables = ["gender","age","Attractiveness_first_impression_mean","education"]
	print("------\tmodel\t------\tfactor\t------\tgender\t------\tage\t------\tattra\t------\teduca")
	print("------\t------\t------\t------\t------\t------\t------\t------\t------\t------\t------\t-----")	
	
	for model in models:
		answer_list = "output_data/%s/answers_%s_%s_%s/"%(model,dataset,question_tpye,"infor" if infor else "noninfor")
		df = get_compared_group(dataset,question_tpye,model,infor,answer_list,select_col_pre,select_col_tru,select_variables)
		n= len(select_col_pre)*2
		df_ = df[df.columns[0:n]]
		df_norm = (df_ - df_.mean()) / (df_.std())
		df_norm_ = (df_ - df_.mean()) / (df_.std())
		df_norm = pd.concat([df_norm_,df[df.columns[n:]]],axis=1)
		#pyreadstat.write_sav(df_norm, "statistical_analysis/%s_%s_%s.sav"%(model,question_tpye,"infor" if infor else "noninfor"))
		columns = df_norm.columns
		for c in range(2):
			print('------\t%s\t'%model,end='')
			print('------\t%s\t'%columns[c][0:6],end='')
			for v in range(4):
				co = corr_df(df_norm,columns[c],columns[v+4])
				print('------\t%.3f\t'%co,end='')
			print("\n------\t------\t------\t------\t------\t------\t------\t------\t------\t------\t------\t------")	
	for i in range(len(list(sc[0]))):
		annotators = ["SBERT","Human","Self"]
		answer_list = "output_data/%s/answers_%s_%s_%s/"%(model,dataset,question_tpye,"infor" if infor else "noninfor")		
		df = get_compared_group(dataset,question_tpye,models[0],infor,answer_list,select_col_pre,list(sc[0])[i],select_variables)
		n= len(select_col_pre)*2
		df_ = df[df.columns[0:n]]
		df_norm = (df_ - df_.mean()) / (df_.std())
		df_norm_ = (df_ - df_.mean()) / (df_.std())
		df_norm = pd.concat([df_norm_,df[df.columns[n:]]],axis=1)
		columns = df_norm.columns
		for c in range(2):
			print('------\t%s\t'%annotators[i],end='')
			print('------\t%s\t'%columns[c][0:6],end='')
			for v in range(4):
				co = corr_df(df_norm,columns[c+2],columns[v+4])
				print('------\t%.3f\t'%co,end='')
			print("\n------\t------\t------\t------\t------\t------\t------\t------\t------\t------\t------\t------")		
	os.system('pause')
	
if __name__ == "__main__":
	main()
	
	
	
	
	
	
	
	