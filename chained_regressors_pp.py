# Andrew Humphrey (2021)

import time
import pickle
import sys
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    median_absolute_error,
	)

from catboost import ( 
    CatBoostRegressor, 
    Pool,
	)

import dask
import dask.dataframe as dd

import warnings
warnings.filterwarnings('ignore')

# define some preprocessing functions and chuck them into a dict
def drop_neq(df,col,value):
	return df[df[col] != value]

def drop_geq(df,col,value):
	return df[df[col] < value]

def drop_leq(df,col,value):
	return df[df[col] > value]

funct_dict = {
	'drop_neq': drop_neq,
	'drop_geq': drop_geq, 
	'drop_leq': drop_leq,
	}

# start timer
t0 = time.time()

# read yaml file from cmd line
yaml_input = sys.argv[1]
with open(yaml_input, "r") as f:
    params_dict = yaml.safe_load(f)

# unpack parameters
metadata = params_dict['data_options']
output_tag = params_dict['output_tag']
test_size = params_dict['test_size']
n_iter = params_dict['n_iter']
calculate_weights = params_dict['calculate_weights']
estimate_uncertainty = params_dict['estimate_uncertainty']
save_output = params_dict['save_output']
output_filename = params_dict['output_filename']
zbins_dict = params_dict['zbins_dict']
wvals = params_dict['wvals']
task_type = params_dict['task_type']
thread_count = params_dict['thread_count']
max_depth_simple = params_dict['max_depth_simple']
n_estimators_simple = params_dict['n_estimators_simple']
max_depth_complex = params_dict['max_depth_complex']
n_estimators_complex = params_dict['n_estimators_complex']
random_state = params_dict['random_state']


# training on GPU with objective=MAE gives a warning about MAE not implemented on GPU
mae_or_rmse = "RMSE" if task_type == "GPU" else "MAE"

# define models
simple_model =  CatBoostRegressor(
	task_type = task_type, 
	logging_level='Silent',
	thread_count=thread_count,
	max_depth=max_depth_simple,
	n_estimators=n_estimators_simple,
	random_state=random_state
	)

complex_model_mape =  CatBoostRegressor(
	task_type = task_type, 
	logging_level='Silent',
	thread_count=thread_count,
	max_depth=max_depth_complex,
	n_estimators=n_estimators_complex,
	objective='MAPE',
	random_state=random_state)

complex_model_mae =  CatBoostRegressor(
	task_type = task_type, 
	logging_level='Silent',
	thread_count=thread_count,
	max_depth=max_depth_complex,
	n_estimators=n_estimators_complex,
	objective=mae_or_rmse,
	random_state=random_state)

# read data into Dask dataframe
infile = params_dict['data_file']
print('\n*** reading file: '+infile+' ***\n')
df = dd.read_parquet(infile,engine='pyarrow')

# create index column called ID
# create new if no index column specified
if metadata['index'] == None:
	df['ID'] = df.index.values
else: # or rename specified index col to ID
	df = df.rename(columns={metadata['index']: 'ID'})

# update metadata
metadata['index'] = 'ID'

# trim unneeded columns
df = df[[metadata['index']]+metadata['features']+metadata['targets']]

# apply conditional cuts using functions and params specified in input
if 'conditional_cuts' in metadata.keys():
	for key in metadata['conditional_cuts'].keys():
		item = metadata['conditional_cuts'][key]
		func = funct_dict[item[0]]
		df = func(df,item[1],item[2])

# rename cols
if 'rename' in metadata.keys():
	df = df.rename(columns=metadata['rename'])

# log10 transform columns if specified in user input
if 'to_log' in metadata.keys():
	if len(metadata['to_log']) > 0:
		for c in metadata['to_log']:
			df[c] = np.log10(df[c])

# undersample at random if sampled_fraction < 1
if metadata['sampled_fraction'] < 1.0:
	df = df.sample(frac=metadata['sampled_fraction'])

# drop cases with nondetection in Y,J or metadata
if 'require_detection' in metadata.keys() and 'missing' in metadata.keys():
	require_detection = metadata['require_detection']
	missing = metadata['missing']
	df[require_detection] = df[require_detection].replace(missing,np.nan)
	df = df.dropna()

# make magnitude cuts if necessary
if 'mag_cuts' in metadata.keys():
	mag_cuts = metadata['mag_cuts']
	for band in mag_cuts:
		df = df[df[band] <= mag_cuts[band]]

# make S/N cut for different filters
# assumes faintest mag = 3 sigma limit
if 'snr_cuts' in metadata.keys():
	snr_cuts = metadata['snr_cuts']
	for band in snr_cuts:
		snr = snr_cuts[band]
		if snr > 3.:
			lim = df[band].compute().max()
			df = df[df[band] <= lim - np.log10(snr/3.)]


# print number of rows in the resulting dataset
print(len(df),'rows of data')

# feature engineering
bands = [x for x in metadata['features'] if x[0]!='d'] # modify as needed
colours=[]
n=len(bands)

for m1 in range(n-1):
	for m2 in range(n-1):
		m2 += 1
		if m2 > m1:
			colour = bands[m1]+'-'+bands[m2]
			df[colour] = df[bands[m1]] - df[bands[m1+1]]
			colours.append(colour)

# define column lists
cols = [metadata['index']] + metadata['features'] + colours + metadata['targets']
cols_ml = metadata['features'] + colours
cols_notargets =  [metadata['index']] + metadata['features'] + colours
cols_targets = metadata['targets']

#update the lists above form changed column names
if 'rename' in metadata.keys():
	for key,value in metadata['rename'].items():
		cols = [x if x != key else value for x in cols]
		cols_ml = [x if x != key else value for x in cols_ml]
		cols_notargets = [x if x != key else value for x in cols_notargets]
		cols_targets = [x if x != key else value for x in cols_targets]

# downcast float64 values to float32
df[cols_ml+cols_targets] = df[cols_ml+cols_targets].astype('float32')

# do z+1 if redshift label exists
z_additive = 1
try: df['redshift'] += z_additive
except: pass

# compute the data transformations and convert to Pandas DataFrame
df =  pd.DataFrame(df[cols].compute())

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[cols_ml],
													df[cols_targets],
                                                    test_size=test_size,
                                                    random_state=random_state)

# generate series with index or indices to allow source ID matching
try:
	test_indices = df[[metadata['index']]]
except:
	index_names = [x for x in df.columns.values.tolist() if 'ID' in x]
	test_indices = df[index_names]

test_indices = test_indices[test_indices.index.isin(X_test.index)]

# print size of train and test sets
print('train:',len(y_train))
print('test:',len(y_test))


# remove the mean and scale column values to unit variance
print('Scaling ...')
scaler = StandardScaler()
scaler.fit(X_train[cols_ml])
X_train[cols_ml] = scaler.transform(X_train[cols_ml])
X_test[cols_ml] = scaler.transform(X_test[cols_ml])

ycols = y_train.columns.values.tolist()
n_outputs = len(ycols)


# before training the chained regressors, first train single models in isolation
# (to see that the chained regression approach gives better results than a single model)
print('Training single models (for comparison with later results) ...')
for target in ycols:
	if target == 'redshift':
		complex_model_mape.fit(X_train[cols_ml],y_train[target])
		preds_target = complex_model_mape.predict(X_test[cols_ml])
		nmad = 1.48 * np.median(np.abs(y_test[target]-preds_target)/(1+y_test[target]-z_additive))
		print('Single regressor for', target, np.round(nmad,6))
	else:
		complex_model_mae.fit(X_train[cols_ml],y_train[target])
		preds_target = complex_model_mae.predict(X_test[cols_ml])
		nmad = 1.48 * np.median(np.abs(y_test[target]-preds_target))
		print('Single regressor for', target, np.round(nmad,6))


# split train into two "folds" to allow OOF prediction to be obtained
X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train,test_size=0.5)

wvals = wvals[::-1]

# function to optimize weights
def optmize_weights(weight_array,targ,zbins_list):
	for i in range(len(weight_array)):
		nmad_w_array = []

		for l,wval in enumerate(wvals):
			weight_array[i] = wval


			weights = y_train1[targ]*0.
			for k,b in enumerate(zbins_list):
				weights[(y_train1[targ] >= b[0]) & (y_train1[targ] < b[1])] = weight_array[k]

			train_pool_w = Pool(data=X_train1,
							label=y_train1[targ],
							weight=weights)

			model = simple_model
			model.fit(train_pool_w)
			preds_w = model.predict(X_train2)

			if 'redshift' in targ:
				nmad_w = 1.48 * np.median(np.abs(y_train2[targ]-preds_w)/(1+y_train2[targ]-z_additive))
			else:
				nmad_w = 1.48 * np.median(np.abs(y_train2[targ]-preds_w))

			nmad_w_array.append(nmad_w)

		weight_array[i] = wvals[np.argmin(nmad_w_array)]

	print('final weights:',weight_array)
	return weight_array

weights_dict = {}
for targ in metadata['weight_targets']:

	if calculate_weights == True:
		print('Optimizing weights for',targ,'...')

		zbins_list = zbins_dict[targ]
		weight_array = np.ones(len(zbins_list))/2.

		weight_array = optmize_weights(weight_array,targ,zbins_list)

		# generate weights1 and weights2 for train1 and train2
		weight_array_best = weight_array

		weights1 = y_train1[targ]*0.
		weights2 = y_train2[targ]*0.

		for k,b in enumerate(zbins_list):
			weights1[(y_train1[targ] >= b[0]) & (y_train1[targ] < b[1])] = weight_array_best[k]
			weights2[(y_train2[targ] >= b[0]) & (y_train2[targ] < b[1])] = weight_array_best[k]

	else:
		weights1 = (y_train1[targ]*0.)+1
		weights2 = (y_train2[targ]*0.)+1

	# store the resulting weight arrays in the dictionary
	weights_dict[targ] = {'weights1':weights1, 'weights2':weights2}


#### next: the actual model training
print('\n*** starting model training ***\n')
print('Training features:\n',X_train1.columns.values)

t_inference = 0 # inference timing


for j in range(n_iter):
	print('\nIteration #',j+1,' of ',n_iter)

	oof1 = np.zeros(shape=(len(X_train1),n_outputs))
	oof2 = np.zeros(shape=(len(X_train2),n_outputs))
	preds = np.zeros(shape=(len(X_test),n_outputs))

	dict_scores = {}
	dict_bias = {}

	if j == 0:
		dict_scores_it0 = {}
		dict_bias_it0 = {}

	preds_temp = pd.DataFrame(columns=ycols)
	for i,name in enumerate(ycols):
		if j==0 and name=='redshift':
			model = complex_model_mape
		else:
			model = complex_model_mae
		key = type(model).__name__

		# extract weight arrays from dictionary
		# 1. check if weights were calculated
		if calculate_weights == True:
			if name in weights_dict.keys():
				weights1 = weights_dict[name]['weights1']
				weights2 = weights_dict[name]['weights2']
		# 2. if no weights exist for the target, use 1
		else:
			weights1 = (y_train1[name]*0)+1
			weights2 = (y_train2[name]*0)+1

		# construct train pool (1)
		train_pool1 = Pool(data=X_train1,
						label=y_train1[name],
						weight=weights1)


		model.fit(train_pool1)
		oof2[:,i] = model.predict(X_train2)
		t_ = time.time() # inference timing
		preds[:,i] = model.predict(X_test)/2.
		t_inference += (time.time() - t_) # inference timing

		# construct train pool (2)
		train_pool2 = Pool(data=X_train2,
							label=y_train2[name],
							weight=weights2)

		model.fit(train_pool2)
		oof1[:,i] = model.predict(X_train1)
		t_ = time.time() # inference timing
		preds[:,i] += model.predict(X_test)/2.
		t_inference += (time.time() - t_) # inference timing

		preds_temp[name] = preds[:,i]

		# check if the target label is redshift and use appropriate metric formulae
		if name == 'redshift':
			# calculate the standard photo-z metrics
			score = 1.48 * np.median(np.abs(y_test[name]-preds[:,i])/(1+y_test[name]-z_additive))
			bias = np.median((preds[:,i]-y_test[name])/(y_test[name]))
			f_outl = len(y_test[np.abs(y_test[name]-preds[:,i])/y_test[name] > 0.15])/len(y_test)

			# calculate the variant of NMAD used by Desprez et al. (Euclid photo-z challenge)
			score_euclid_pz_challenge = 1.48 * np.median(np.abs((y_test[name]-preds[:,i])/(1+y_test[name]-z_additive) - (score/1.48) ) )

			scorename='NMAD'
			print('('+name+')',' NMAD: ',np.round(score,5),' Outlier fr: ',np.round(f_outl,5),' Bias:',np.round(bias,5),
					'Euclid phz chal. NMAD:',np.round(score_euclid_pz_challenge,5))
		
		# otherwise, calculate metrics using the general formulae
		else:
			score = 1.48 * np.median(np.abs(y_test[name]-preds[:,i]))
			bias = np.median((preds[:,i]-y_test[name]))
			f_outl = len(y_test[np.abs(y_test[name]-preds[:,i]) > 0.3])/len(y_test)
			mae = mean_absolute_error(y_test[name],preds[:,i])
			r2 = r2_score(y_test[name],preds[:,i])
			scorename='NMAD'

			print('('+name+')',' NMAD: ',np.round(score,5),' Outlier fr: ',np.round(f_outl,5),' Bias:',np.round(bias,5),
				'Mean Abs. Err:',np.round(mae,5),'R2:',np.round(r2,5))


		f_outl_str=str(f_outl)[:5]
		if j==0: f_outl_str_it0=str(f_outl)[:5]


		dict_scores[name] = score
		dict_bias[name] = bias

		if j == 0:
			dict_scores_it0[name] = score
			dict_bias_it0[name] = bias

	# place the OOF features into new dataframes ready to concat with the original features
	newcols = [x+'_it'+str(j)+'_'+key for x in ycols]
	if j == 0: newcols_it0 = newcols # make copy of colnames of it0 output
	oof1 = pd.DataFrame(oof1,index=X_train1.index,columns=newcols)
	oof2 = pd.DataFrame(oof2,index=X_train2.index,columns=newcols)
	preds = pd.DataFrame(preds,index=X_test.index,columns=newcols)

	# concatenate OOF columns into the training or test set features
	X_train1 = pd.concat([X_train1,oof1],axis=1)
	X_train2 = pd.concat([X_train2,oof2],axis=1)
	X_test = pd.concat([X_test,preds],axis=1)

	# create temp df to show output together with target labels
	y_test_df = pd.DataFrame(y_test,index=X_test.index)
	test_output = pd.concat([X_test[newcols],y_test_df],axis=1)
	if j == 0: test_output_it0 = pd.concat([X_test[newcols],y_test_df],axis=1)

	# save column names for further use
	last_cols = newcols
	last_ycols = ycols



if estimate_uncertainty == True:

	# train regressor model to predict residuals
	print('estimating uncertainties by predicting the residuals ...')

	X_train1_res = X_train1.copy()
	X_train2_res = X_train2.copy()
	X_test_res = X_test.copy()
	mlcols = X_train1_res.columns.values.tolist()
	rescols = []

	for i, ycol in enumerate(last_ycols):
		itercol = last_cols[i]
		res1 = np.abs(y_train1[ycol]-X_train1_res[itercol])
		res2 = np.abs(y_train2[ycol]-X_train2_res[itercol])
		X_train1_res[ycol+'_residual'] = res1
		X_train2_res[ycol+'_residual'] = res2
		rescols.append(ycol+'_residual')

		if 'redshift' in ycol:
			ycol_redshift = ycol
			pred_col_redshift = itercol


	X_train_res = pd.concat((X_train1_res,X_train2_res))
	X_train_res.reset_index(inplace=True)
	y_train_res = pd.concat((y_train1,y_train2))
	y_train_res.reset_index(inplace=True)

	# train a regressor to predict the residuals
	model = CatBoostRegressor(logging_level='Silent',
						   thread_count=thread_count,
						   max_depth=5,
						   n_estimators=500,
						   objective='Poisson')
	
	for rescol in rescols:
		model.fit(X_train_res[mlcols],X_train_res[rescol])
		preds_res = model.predict(X_test)
		X_test_res[rescol] = preds_res


# saving the results to file
if save_output == True:
	results_dict = {'method':'catboost',
					'results':test_output.sort_index(),
					'uncertainties':X_test_res.sort_index(),
					'iter':n_iter-1,
					'features':X_test[cols_ml].sort_index(),
					'indices':test_indices.sort_index(),
					}

	if output_filename == None:
		with open('output'+output_tag+'.pickle', 'wb') as handle:
			pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	elif isinstance(output_filename, str):
		with open(output_filename+output_tag+'.pickle', 'wb') as handle:
			pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# calculate running times and print results to terminal
t1= time.time()-t0
elapsed = t1
units = 's'
if t1 > 60:
	elapsed = t1/60
	units = 'm'

if t1 > 3600:
	elapsed = t1/3600
	units = 'h'

print('Elapsed time:',np.round(elapsed,3),units)
print('Inference only:',t_inference,'s')
print('Inference time per galaxy:',(t_inference/len(X_test)),'s')

quit()
