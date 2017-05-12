import numpy as np
from scipy.stats import norm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
import allensdk.brain_observatory.stimulus_info as stim_info
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from scipy import asarray as ar,exp
from scipy import stats
from scipy.stats import spearmanr
from scipy.stats import levene
import pandas as pd
import sys
import logging
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from statsmodels.robust.scale import mad
import cv2
from scipy.ndimage.filters import gaussian_filter
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.qda import QDA
from sklearn.svm import SVC


logging.basicConfig()

def generate_data(T = 10000, n = 30, eps = 1e-4, 
				noise_model = 'exponential', non_lin = np.exp, 
				c = 3, scale = 5, filt_amp = 10, stim = None):
	'''
	currently supports noise_model = {'exponential', 'gaussian', 'poisson'}
	non_lin should be any function that applies elementwise to it's single 
	argument. 

	returns design, weights, observations
	'''

	if noise_model == 'exponential':
		stim, weights, y = generate_gamma_data(non_lin, T = T, n = n, eps = eps,
											c = c, scale = scale, filt_amp = filt_amp, stim = stim)

	elif noise_model == 'gaussian':
		stim, weights, y = generate_gaussian_data(non_lin, T = T, n = n, eps = eps,
		 									c = c, scale = scale, filt_amp = filt_amp, stim = stim)

	elif noise_model == 'poisson':
		stim, weights, y = generate_poisson_data(non_lin, T = T, n = n, eps = eps,
		 									c = c, scale = scale, filt_amp = filt_amp, stim = stim)
	elif noise_model == None:
		stim, weights, y = generate_gaussian_data(non_lin, T = T, n = n, eps = eps,
									c = c, scale = scale, filt_amp = filt_amp, stim = stim)
		
		y = non_lin(stim.dot(weights))

	return stim, weights, y


def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x)) + 1e-4

def tf_soft_rec(x):
	return tf.log(1 + tf.exp(x))
def np_soft_rec(x):
	return np.log(1 + np.exp(x))

def cond_int(non_lin, weights, stim, scale, c, nls = 0):
	'''
	returns the conditional intensity \lambda = f(w^TX)
	'''
	h = stim.dot(weights)
	f = scale*(non_lin(h-c) + nls)
	return f

def gamma_model(cond_int, p = 2):
	'''
	draws from a gamma distribution with shape parameter p. 
	and mean 'cond_int'
	'''
	y = np.random.gamma(p, cond_int/p)
	return y


def poisson_model(cond_int):
	y = np.random.poisson(cond_int)
	return y

def gaussian_model(cond_int):
	y = np.random.normal(cond_int, 1)
	return y

def generate_gamma_data(non_lin, T = 1000, n = 30, eps = 1e-1, c = 3, scale = 5, filt_amp = 10, stim = None):
	if stim == None:
		stim = np.random.normal(0, scale = 2, size = [T, n])
	weights = filt_amp*norm.pdf(range(0, n), loc = n/2, scale = n/10)
	y = gamma_model(cond_int(non_lin, weights, stim, scale, c) + eps)
	return stim, weights, y

def generate_poisson_data(non_lin, T = 1000, n = 30, eps = 1e-1, c = 3, scale = 5, filt_amp = 10, stim = None):
	'''
	poisson data with any non-linearity
	'''
	if stim == None:
		stim = np.random.normal(0, scale = 2, size = [T, n])
	weights = filt_amp*norm.pdf(range(0, n), loc = n/2, scale = n/10)
	y = poisson_model(cond_int(non_lin, weights, stim, scale, c))
	return stim, weights, y

def generate_gaussian_data(non_lin, T = 1000, n = 30, eps = 1e-1, c = 3, scale = 5, filt_amp = 10, stim = None):
	'''
	sigmoidal non-linearity
	'''
	if stim == None:
		stim = np.random.normal(0, scale = 2, size = [T, n])
	
	weights = filt_amp*norm.pdf(range(0, n), loc = n/2, scale = n/10)
	y = gaussian_model(cond_int(non_lin, weights, stim, scale, c))
	return stim, weights, y

def gridplot(num_rows, num_cols):
	'''get axis and gridspec objects for grid plotting 
	returns gs, ax
	'''
	gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.0)
	ax = [plt.subplot(gs[i]) for i in range(num_rows*num_cols)]

	return gs, ax

def simpleaxis(ax, bottom = False):
	'''
	remove the top and right spines and ticks from the axis. 
	'''
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	if bottom:
		ax.spines['bottom'].set_visible(False)
		ax.get_xaxis().set_ticks([])
	else:
		ax.get_xaxis().tick_bottom()
	

	ax.get_yaxis().tick_left()

def relu(X):
	return X*(X > 0)

def get_data_stats(all_tensors):
	'''
	returns mean, std 
	'''
	mean = [0]
	std = [0]

	for key in all_tensors.keys():
	    
	    #this computes the mean across the axis of numTrials.
	    t_mean = np.mean(all_tensors[key], 2).flatten()
	    t_std = np.std(all_tensors[key], 2).flatten()
	    
	    
	    mean = np.concatenate((mean, t_mean))
	    std = np.concatenate((std, t_std))
	
	return mean, std

def sort_scores(scores_dict):

	t_per_exp = []
	t_per_gaus = []

	t_per_l_exp = []
	t_per_l_sig = []
	t_per_l_sr = []

	le_dict = {}
	lg_dict = {}


	for key in scores_dict.keys():
	    
	    scores, features = scores_dict[key]
	    
	    n_cells, n_nl, n_nm = scores.shape

	    best_noise_model = []
	    best_non_linearity = []

	    likelihood_exponential = []
	    likelihood_gaussian = [] 

	    for i in range(n_cells):
	        idx = np.argmin(scores[i])

	        nl_ind, nm_ind = np.unravel_index(idx, (n_nl, n_nm))

	        if scores[i, nl_ind, nm_ind] != np.nan:
	            best_noise_model.append(nm_ind)
	            best_non_linearity.append(nl_ind)


	            if nm_ind == 0:
	                likelihood_exponential.append(scores[i, nl_ind, nm_ind])

	            if nm_ind == 1:
	                likelihood_gaussian.append(scores[i, nl_ind, nm_ind])
	                
	    best_noise_model = np.array(best_noise_model)
	    best_non_linearity = np.array(best_non_linearity)
	    
	    per_exp = sum(best_noise_model == 0) / float(len(best_noise_model))
	    per_gaus = sum(best_noise_model == 1) / float(len(best_noise_model))
	    
	    per_l_exp = sum(best_non_linearity == 0) / float(len(best_non_linearity))
	    per_l_sig = sum(best_non_linearity == 1) / float(len(best_non_linearity))
	    per_l_sr = sum(best_non_linearity == 2) / float(len(best_non_linearity))
	    
	    t_per_exp.append(per_exp)
	    t_per_gaus.append(per_gaus)
	    
	    t_per_l_exp.append(per_l_exp)
	    t_per_l_sig.append(per_l_sig)
	    t_per_l_sr.append(per_l_sr)

	    
	    le = -np.array(scores[:, :, 0].flatten())
	    le = le[~np.isnan(le)] 
	    lg = -np.array(scores[:, :, 1].flatten())
	    lg = lg[~np.isnan(lg)]

	    le_dict[key] = le
	    lg_dict[key] = lg


	return t_per_exp, t_per_gaus, t_per_l_exp, t_per_l_sig, t_per_l_sr, lg_dict, le_dict

def get_explainable_variance(data_tensor):
	'''
	data_tensor should be the output of arrange_data_trialTensor
	
	computes explainable variance by randomly splitting the trials in half, and regressing 1/2 the trials against
	the other half of the trials. 
	'''
	from scipy.stats import linregress

	n_neurons, n_conditions, n_trials, trialLength = data_tensor.shape

	data_tensor = np.mean(data_tensor, axis = 3)

	trial_idx = np.arange(0, n_trials)
	trial_idx = np.random.permutation(trial_idx)

	tr_1 = trial_idx[0:n_trials/2]
	tr_2 = trial_idx[n_trials/2:]

	av_1 = np.mean(data_tensor[:, :, tr_1], axis = 2)
	av_2 = np.mean(data_tensor[:, :, tr_2], axis = 2)

	ra_variance = np.zeros([n_neurons])

	for i in range(n_neurons):
		slope, intercept, r_value, p_value, std_err = linregress(av_1[i], av_2[i])
		ra_variance[i] = r_value**2

	return ra_variance

def download_data(region, cre_line, stimulus = None):
	'''
	region = [reg1, reg2, ...]
	cre_line = [line1, line2, ...]
	'''
	boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
	ecs = boc.get_experiment_containers(targeted_structures=region, cre_lines=cre_line)

	ec_ids = [ ec['id'] for ec in ecs ]

	exp = boc.get_ophys_experiments(experiment_container_ids=ec_ids)

	if stimulus == None:
		exp = boc.get_ophys_experiments(experiment_container_ids=ec_ids)

	else:
		exp = boc.get_ophys_experiments(experiment_container_ids=ec_ids, stimuli = stimulus)


	exp_id_list = [ec['id'] for ec in exp]

	data_set = {exp_id:boc.get_ophys_experiment_data(exp_id) for exp_id in exp_id_list}

	return data_set 

def pca_features(images, scale = False):
	'''
	Can pass images before or after re-shaping. Should pass before reshaping if scale = True
	returns: scenes_r, basis <-- the dimensionality reduced images, and the basis vectors. 
	'''

	if scale == True:

		av = np.mean(images, axis = 0)
		std = np.std(images, axis = 0)

		images -= av
		images /= std

	model = PCA()
	scenes_r = model.fit_transform(images.reshape([len(images), -1])) 

	basis = model.components_
	return scenes_r, basis

def ica_features(images, scale = False):
	if scale == True:

		av = np.mean(images, axis = 0)
		std = np.std(images, axis = 0)

		images -= av
		images /= std

	model = FastICA()
	scenes_r = model.fit_transform(images.reshape([len(images), -1])) 

	basis = model.components_
	return scenes_r, basis		
def get_data(data_set, stimulus):
	'''
	returns dff, images, stim_table
	'''

	time, dff_traces = data_set.get_dff_traces()

	try:
		images = data_set.get_stimulus_template(stimulus)
	except:
		print "No stimulus template..."
		images = None

	stim_table = data_set.get_stimulus_table(stimulus)

	return dff_traces, images, stim_table


def arrange_data_tuning(dff, dxcm, stim_table, ratio = True, fps = 30):
	'''
	Calculate average responses to different stimulus conditions, with the stim_table serving
	as a lookup table. 

	If ratio is True, response will be calculated in the AIBS way, average(stim_onset + .5 sec)
	/average(stim_onset - .5 secs)

	returns responses: n_conditions x n_neurons
	'''

	responses = []

	for i, row in stim_table.iterrows():

		if ratio:
			baseline = np.average(dff[:, row['start'] - fps : row['start']], axis = 1)
			response = np.average(dff[:, row['start'] + 2 : row['start'] + 2*fps ], axis = 1)

			ev_resp = (response - baseline) / baseline
		else:
			ev_resp = np.average(dff[:, row['start'] + 2: row['start'] + 2*fps], axis = 1)

		responses.append(ev_resp)

	return np.array(responses)



def arrange_data_glm(dff_traces, images, stim_table):
	#declare a dictionary of empty lists for each cell trace, 
	#and a list for the stimulus
	data = []
	stim_array = []
	im_array = []

	#average each trace over the presentation of each stimulus
	for index, row in stim_table.iterrows():
	    stim_array.append(images[row['frame']])
	    im_array.append(row['frame'])

	    try:
	    	data.append(np.mean(dff_traces[:, row['start'] + 6 :row['start'] + 9], axis = 1)) #/ np.mean(dff_traces[:, row['start'] : row['start'] - 30], axis = 1) )
	    except:
	    	data.append(np.mean(dff_traces[:, row['start']+ 6:row['end']], axis = 1))
	stim_array = np.array(stim_array)
	#stim_array = stim_array[:, 0:10]

	data = np.array(data)

	return data, stim_array, im_array

def arrange_data_glm_temporal(dff_traces, images, stim_table):
	'''
	use a temporal basis vector to create features for each dff observation. 
	'''
	data = []
	stim_array = []

	prev_row = None;
	for index, row in stim_table.iterrows():
		for i, ind in enumerate(range(row['start'], row['end'])):
			data.append(dff_traces[:, ind])

			try:
				if prev_row == None:
					stim_array.append(images[row['frame']])
			except:
				stim_array.append((images[prev_row['frame']] + i*images[row['frame']])/ float(i + 1))

		prev_row = row

	return np.array(data), np.array(stim_array)


def arrange_rs_glm(rs, stim_table):

	running_speed = []

	for index, row in stim_table.iterrows():
		running_speed.append(np.average(rs[row['start']:row['end']]))

	return running_speed

def get_index_array(stim_table):
	index_array = []

	for index, row in stim_table.iterrows():
		index_array.append(row['frame'])
	return np.array(index_array)

def trial_average(data, index_array):

	l, n_neurons = data.shape

	n_conditions = len(set(index_array))


	trial_average = np.zeros([n_conditions, n_neurons])
	trial_std = np.zeros([n_conditions, n_neurons])

	cond_indices = np.zeros(n_conditions)

	for i in range(l):
		trial_average[index_array[i]] += data[i]
		cond_indices[index_array[i]] += 1


	trial_average /= cond_indices[:, np.newaxis]

	for i in range(l):
		trial_std[index_array[i]] += (data[i] - trial_average[index_array[i]])**2

	trial_std /= cond_indices[:, np.newaxis]

	trial_std = np.sqrt(trial_std) #/ np.sqrt(cond_indices[:, np.newaxis])

	return trial_average, trial_std

def filter_by_exp_variance(dataset, var_thresh = .10, filt_cells = None):
	'''
	Hardcoded for natural scenes data right now

	returns: pruned_dff_traces, images, stim_table
	'''
	dff, images, stim_table = get_data(dataset, stim_info.NATURAL_SCENES)

	if filt_cells != None:
		dff = dff[filt_cells]

	data_tensor, trialCount = arrange_ns_data_trialTensor(dff, stim_table)
	ra_variance = get_explainable_variance(data_tensor)

	exp_ind = ra_variance > var_thresh

	print "neurons with >", 100 * var_thresh, "% explainable variance for dataset  : ", sum(exp_ind)

	pruned_dff_traces = dff[exp_ind]

	return pruned_dff_traces, images, stim_table

def filtered_resized_features(images, gauss_kernel = 1, resize_dim = (256, 256)):


	r_images = []

	for image in images:
		im = cv2.resize(image, resize_dim)
		im = gaussian_filter(im, sigma = gauss_kernel, mode = 'constant', cval = 0)
		r_images.append(im)
	    
	r_images -= np.mean(r_images, axis = 0)
	r_images /= np.std(r_images, axis = 0)

	r_images = np.array(r_images).reshape(len(r_images), -1)

	return r_images

def on_off_features(images):

        
        r_images_pos = np.abs(r_images > 0)
        r_images_neg = np.abs(r_images < 0)



def arrange_ns_data_trialTensor(dff_traces, stim_table, offset = 6, lag = 9):
	'''
	In this function we want to take dff traces (unarranged)
	and return data_Tensor = n_neurons x n_conditions x n_trials x trialLength

	This is helpful for computing statistics of the data, like mean vs. standard deviation, 
	and could be a useful multipurpose pipelining tool in the future.   
	
	at this point this is untested with stim_table objects other than that from 'Natural Scenes'

	'''

	n_neurons, _ = dff_traces.shape
	trialLength = lag - offset
	n_conditions = len(stim_table['frame'].unique())
	n_trials = np.floor(len(stim_table['frame']) / n_conditions)

	data_Tensor = np.zeros([n_neurons, n_conditions, n_trials, trialLength]);

	trialCount = np.zeros([n_conditions])
	for index, row in stim_table.iterrows():
		data_Tensor[:, row['frame'], trialCount[row['frame']], :] = dff_traces[:, row['start'] + offset: row['start'] + lag]
		trialCount[row['frame']] +=1

	return data_Tensor, trialCount



def arrange_data_rs(data_set, bin = True):
	'''
	arranges the data for running speed tuning curve purposes
	dataset = {experiment_id: boc dataset}
	'''
	ds_data = {}

	#collect the stimulus tables, and running speed for each dataset
	for ds in data_set.keys():
	    _, dff = data_set[ds].get_dff_traces()
	    cells = data_set[ds].get_cell_specimen_ids()
	    
	    data = {'cell_ids':cells, 'raw_dff':dff }
	    for stimulus in data_set[ds].list_stimuli():
	        
	        if stimulus == 'spontaneous':      
	            table = data_set[ds].get_spontaneous_activity_stimulus_table()
	        else:
	            table = data_set[ds].get_stimulus_table(stimulus)
	            
	        data[stimulus] = table

	    dxcm, dxtime = data_set[ds].get_running_speed()
	    data['running_speed'] = dxcm
	    
	    ds_data[ds] = data

	#arrange the data for each separate stimuli in a dictionary. Not averaging over
	#presentation of a given image, just concatenating all cell traces, and corresponing
	#running speed. 
	arranged_data = {}
	for ds in data_set.keys():
	    dff_data = ds_data[ds]
	    
	    data = {}
	    for stimulus in data_set[ds].list_stimuli():
	        rs = np.zeros([1])
	        dfof = np.zeros([len(dff_data['cell_ids']), 1])
	        for index, row in dff_data[stimulus].iterrows():
	            dfof = np.concatenate((dfof, dff_data['raw_dff'][:, row['start']: row['end']]), axis = 1)
	            rs = np.concatenate((rs, dff_data['running_speed'][row['start']: row['end']]), axis = 0)     

	        data[stimulus + '_rs'] = np.array(np.squeeze(rs))
	        data[stimulus + '_dff'] = np.array(np.squeeze(dfof))
	        
	    arranged_data[ds] = data  

	#groups the data into 'natural', 'spontaneous', or 'artificial'
	#TODO: subsample

	tb_data = {}
	for ds_id in arranged_data.keys():
	    
	    data  = arranged_data[ds_id]
	    
	    #binning into synthetic, natural, and stimulus
	    _data = {'synthetic_rs': None, 'natural_rs': None, 'spontaneous_rs': None,'synthetic_dff': None, 'natural_dff': None, 'spontaneous_dff':None}
	    
	    #just binning into stimulus and spontaneous
	    #_data = {'stimulus_rs': None, 'spontaneous_rs': None,'stimulus_dff': None, 'spontaneous_dff':None}
	    
	    for stimulus in data_set[ds_id].list_stimuli():
	        
	        if (stimulus == 'locally_sparse_noise') or ('gratings' in stimulus):
	            #stim_key = 'stimulus'
	            stim_key = 'synthetic'
	        elif ('natural' in stimulus):
	            #stim_key = 'stimulus'
	            stim_key = 'natural'
	        elif ('spontaneous' == stimulus):
	             stim_key = 'spontaneous'
	          
	        #stim_key = stimulus  
	        run_speed =  np.array(data[stimulus + '_rs'])
	        dff = np.array(data[stimulus + '_dff'])
	        
	        if _data[stim_key + '_rs'] == None:
	            _data[stim_key+ '_rs'] = run_speed
	        else:
	            _data[stim_key + '_rs'] = np.concatenate((_data[stim_key + '_rs'], run_speed), axis = 0)

	           
	        if _data[stim_key + '_dff'] == None:
	            _data[stim_key+ '_dff'] = dff
	        else:
	            _data[stim_key + '_dff'] = np.concatenate((_data[stim_key + '_dff'], dff), axis = 1)		    
	    
	    tb_data[ds_id] = _data  
	
	return tb_data, arranged_data

def make_tuning_curves(tb_data, data_set):
	'''
	returns a nested dictionary with key experiment id, key stimulus name, with 
	a (tuning curve, (rho, spearmansp, levensp)) tuple. Tuning curve is a dictionary with key 
	cell specimen ids, which contains a (19, 4) numpy array. The 0th column is the average response, 
	the 1st column is the standard error, the 2nd column is the average shuffled response, and the 
	3rd column is the shuffled standard error. 
	'''

	rs_results = {}
	bin_hist = np.zeros([19, 2])
	shuf_hist = np.zeros([19, 2])

	for ds in tb_data.keys():
	    stim_results = {}
	    for stimulus in data_set[ds].list_stimuli():
	        
	        if ('gratings' in stimulus) or (stimulus == 'locally_sparse_noise'):
	            #stim_key = 'stimulus'
	            stim_key = 'synthetic'
	        if ('natural' in stimulus):
	            #stim_key = 'stimulus'
	            stim_key = 'natural'
	        if ('spontaneous' == stimulus):
	            stim_key = 'spontaneous'
	        
	        
	        neural_responses = {k: np.ones([19,4]) for k in data_set[ds].get_cell_specimen_ids()}            
	        results = {}
	        run_speed = np.array(tb_data[ds][stim_key + '_rs']).flatten()
	        
	        
	        if max(run_speed) < 15:

	            pass
	        elif sum(run_speed <= .5) > 4*sum(run_speed >.5):
	        	pass
	        else:
	        
	            run_speed_shuffled = np.random.permutation(run_speed)
	        
	            bins = stats.mstats.mquantiles(run_speed, np.linspace(0, 1, 20), limit = (0, 50))
	            shuf_bins = stats.mstats.mquantiles(run_speed_shuffled, np.linspace(0, 1, 20), limit = (0, 50))
	            
	            for ind, k in enumerate(data_set[ds].get_cell_specimen_ids()):

	                temp = np.array(tb_data[ds][stim_key + '_dff'][ind])  
	                
	                for i in range(1, len(bins)):
	                    bin_hist[i- 1] = [bins[i -1], bins[i]]
	                    shuf_hist[i-1] = [bins[i-1], bins[i]]
	                    
	                    idx = np.where((run_speed > bins[i-1]) & (run_speed < bins[i]))
	                    shuf_idx = np.where((run_speed_shuffled > shuf_bins[i-1]) & (run_speed_shuffled < shuf_bins[i]))
	                    
	                    #this control shouldn't be necessary
	                    if len(idx[0] != 0):
	                        av = np.mean(temp[idx[0]])
	                        std = np.std(temp[idx[0]])
	                        std_shuf = np.std(temp[shuf_idx[0]])
	                        av_shuf = np.mean(temp[shuf_idx[0]])
	                    else:
	                        av = 0
	                        std = 0

	                    neural_responses[k][i-1, 0] = av
	                    neural_responses[k][i-1, 1] = std / np.sqrt(len(temp[idx[0]]))
	                    neural_responses[k][i-1, 2] = av_shuf
	                    neural_responses[k][i-1, 3] = std / np.sqrt(len(temp[shuf_idx[0]]))

	                
	                x = np.log(np.mean(bin_hist, axis = 1))
	                y = np.array(neural_responses[k][:, 0])

	                shuf_x = np.log(np.mean(bin_hist, axis = 1))
	                shuf_y = np.array(neural_responses[k][:, 2])
	                
	                stat, pvalue = levene(y, shuf_y)
	                
	                n = len(x)
	                
	                ymax = max(y)
	                xmax = x[np.where(y == ymax)[0]]
	                
	                rho, p = spearmanr(x, y)
	                    
	                results[k] = rho, p, pvalue

	        stim_results[stim_key] = (neural_responses, results)    
	    rs_results[ds] = stim_results	

	return rs_results


def gaussian_variance(y, x, w, o, nls, s, non_lin = sigmoid):
	T = float(len(y))
	c_int = cond_int(non_lin, w, x, s, o, nls)

	return (1./T * sum((y-c_int)**2))

def mad_scaling(data):

	L, N = data.shape

	offset = np.mean(data, axis = 0) - mad(data, axis = 0, c = .6);
	top = np.mean(data, axis = 0) + mad(data, axis = 0, c = .6);
	return offset, top

def tf_identity(data):
	return data

def fancy_bar_plot(ts1, ts2, title, label1, label2, ts3 = None, label3 = None):

	stat_av = ts1
	run_av = ts2

#	ax = plt.subplot(111)

	plt.scatter(np.ones([len(stat_av)]) + np.random.uniform(-.025, .025, size = len(stat_av)), stat_av, c = 'm', 
	            alpha = .5, s = 15, label = label1)
	plt.plot([.95, 1.05], [np.average(stat_av), np.average(stat_av)], c = 'k', alpha = .9, linewidth = 2)

	plt.plot([1.0, 1.0], [np.average(stat_av) - np.std(stat_av)/np.sqrt(len(stat_av)), 
	                      np.average(stat_av) + np.std(stat_av)/np.sqrt(len(stat_av))], 
	         alpha = .9, c = 'k', linewidth = 2)

	plt.plot([.99, 1.01], [np.average(stat_av) + np.std(stat_av)/np.sqrt(len(stat_av)) + .001, 
	                      np.average(stat_av) + np.std(stat_av)/np.sqrt(len(stat_av)) + .001], 
	         alpha = .9, c = 'k', linewidth = 2)


	plt.plot([.99, 1.01], [np.average(stat_av) - np.std(stat_av)/np.sqrt(len(stat_av)) - .001, 
	                      np.average(stat_av) - np.std(stat_av)/np.sqrt(len(stat_av)) - .001], c = 'k', linewidth = 2)

	plt.scatter(1.3*np.ones([len(run_av)])  + np.random.uniform(-.025, .025, size = len(run_av)), run_av, c = 'b', 
	            alpha = .5, s = 15, label = label2)
	plt.plot([1.25, 1.35], [np.average(run_av), np.average(run_av)], c = 'k', alpha = .75, linewidth = 2)
	plt.plot([1.3, 1.3], [np.average(run_av) - np.std(run_av)/np.sqrt(len(run_av)), 
	                      np.average(run_av) + np.std(run_av)/np.sqrt(len(run_av))], 
	         alpha = .9, c = 'k', linewidth = 2)

	plt.plot([1.29, 1.31], [np.average(run_av) + np.std(run_av)/np.sqrt(len(run_av)) + .001, 
	                      np.average(run_av) + np.std(run_av)/np.sqrt(len(run_av)) + .001], 
	         alpha = .9, c = 'k', linewidth = 2)


	plt.plot([1.29, 1.31], [np.average(run_av) - np.std(run_av)/np.sqrt(len(run_av)) - .001, 
	                      np.average(run_av) - np.std(run_av)/np.sqrt(len(run_av)) - .001], c = 'k', linewidth = 2)



	if ts3 != None:
		plt.scatter(1.6*np.ones([len(ts3)])  + np.random.uniform(-.025, .025, size = len(ts3)), ts3, c = 'c', 
	            alpha = .5, s = 15, label = label3)
		plt.plot([1.55, 1.65], [np.average(ts3), np.average(ts3)], c = 'k', alpha = .75, linewidth = 2)
		plt.plot([1.6, 1.6], [np.average(ts3) - np.std(ts3)/np.sqrt(len(ts3)), 
		                      np.average(ts3) + np.std(ts3)/np.sqrt(len(ts3))], 
		         alpha = .9, c = 'k', linewidth = 2)

		plt.plot([1.59, 1.61], [np.average(ts3) + np.std(ts3)/np.sqrt(len(ts3)) + .001, 
		                      np.average(ts3) + np.std(ts3)/np.sqrt(len(ts3)) + .001], 
		         alpha = .9, c = 'k', linewidth = 2)


		plt.plot([1.59, 1.61], [np.average(ts3) - np.std(ts3)/np.sqrt(len(ts3)) - .001, 
		                      np.average(ts3) - np.std(ts3)/np.sqrt(len(ts3)) - .001], c = 'k', linewidth = 2)



	
#	plt.ylim([0, 2000])
	plt.ylabel(title)
	plt.legend(frameon = False)
	plt.xlim([.8, 2.75])
#	simpleaxis(ax, bottom=True)
#	plt.show()
def project_parallel(b, v, epsilon= 1e-15):
    '''
    Projection of b along v
    
    Inputs:
        - b: 1-d array
        - v: 1-d array
        - epsilon: threshold for filling 0 for squared norms
    
    Output:
        - 1-d array: the projection of b parallel to v
    '''
    sigma = (np.dot(b,v)/np.dot(v,v)) if np.dot(v,v) > epsilon else 0
    return (sigma*v)

def tensorize(data, labels, shuffle = False, zero_mean = False, whiten = False, min_class_els = 0):
    
    labels = labels.flatten()
    n_points, n_neurons = data.shape
    n_classes = len(set(np.array(labels)))
    
    tensor = [np.empty([0, n_neurons]) for i in range(n_classes)]  
    trial_count = np.zeros(n_classes).astype('int')
   
    label_map = {sorted(set(np.array(labels)))[i]:i for i in range(n_classes)}

    #print data.shape
    
    for i in range(n_points):
        if trial_count[label_map[labels[i]]] == 0:
			tensor[label_map[labels[i]]] = data[np.newaxis, i]
        else:
			tensor[label_map[labels[i]]] = np.concatenate((tensor[label_map[labels[i]]], data[np.newaxis, i]), axis= 0 )
        
        trial_count[label_map[labels[i]]] += 1
        
    num_trials = min(trial_count).astype('int')

    #print num_trials
    
    if shuffle:
        
        for i in range(n_neurons):
            for j in range(n_classes):
                tensor[j] = np.array(tensor[j])
                if len(tensor[j]) > 0:
                    tensor[j][:, i] = np.random.permutation(tensor[j][:, i])
       
    
    new_data = np.empty([n_points, n_neurons])
    new_labels = []
    
    k = 0
    for i in range(n_classes):
        j = len(tensor[i])
        if j > min_class_els:
            for l in range(j):
                new_labels.append(i)

            
            #print tensor[i].shape#, k, k + j, new_data[k:k+j].shape
            temp = tensor[i]
            if zero_mean:
                mean = np.mean(tensor[i], axis = 0)
                temp -= mean
            if whiten:
                mean = np.mean(tensor[i], axis = 0)
                for m in range(tensor[i].shape[1]):     
                    temp[:, m] = np.random.normal(mean[m], 0.001, size = temp.shape[0])
                
            
            new_data[k:k+j] = temp
            k += j
    
    new_data = new_data[0:len(new_labels)]
    return new_data, np.array(new_labels), tensor


def cross_validate_decoding(all_run_tensors, all_stat_tensors, model = GaussianNB, whiten = False, tr_shuffle = False, te_shuffle = False,
							cv_folds = 10, min_class_els = 5, nomean = False, shuffle = False):
	run_scores = []
	stat_scores = []

	percent_diff = []

	for key in sorted(all_run_tensors.keys()):
    
	    data_run, trialNum_run = all_run_tensors[key]
	    data_stat, trialNum_stat = all_stat_tensors[key]
	    
	    
	    run_score = []
	    stat_score = []
	    
	    for i in range(cv_folds):
	        
	        
	        data_run, trialNum_run, _ = tensorize(data_run, np.array(trialNum_run).flatten() - 1, min_class_els = min_class_els, shuffle = shuffle)      
	        data_stat, trialNum_stat, _ = tensorize(data_stat, np.array(trialNum_stat).flatten() -1, min_class_els = min_class_els, shuffle = shuffle)
	        
	        rdff_train, rdff_test, rtrialNums_train, rtrialNums_test = train_test_split(data_run, trialNum_run)
	        sdff_train, sdff_test, strialNums_train, strialNums_test = train_test_split(data_stat, trialNum_stat)

	      
#	        rdff_train, rtrain_labels, _ = tensorize(rdff_train, rtrialNums_train, shuffle = tr_shuffle, nomean= nomean, whiten = whiten, min_class_els = 0)
#	        rdff_test, rtest_labels, _ = tensorize(rdff_test, rtrialNums_test, shuffle = te_shuffle, nomean = nomean, whiten = whiten, min_class_els = 0)
	        
#	        sdff_train, strain_labels, _ = tensorize(sdff_train, strialNums_train, shuffle = tr_shuffle, nomean = nomean, whiten = whiten, min_class_els = 0)
#	        sdff_test, stest_labels,_ = tensorize(sdff_test, strialNums_test, shuffle = te_shuffle, nomean = nomean, whiten = whiten, min_class_els = 0)    
	    
	        model = GaussianNB()
	        model.fit(sdff_train, strialNums_train)                   
	        stat_score.append(model.score(sdff_test, strialNums_test))
	        
	        
	        model = GaussianNB()
	        model.fit(rdff_train, rtrialNums_train)
	        run_score.append(model.score(rdff_test, rtrialNums_test))

	    
	    percent_diff.append(2*(np.average(run_score) - np.average(stat_score))/ 
	               (np.average(run_score) + np.average(stat_score)))  
	    run_scores.append(np.average(run_score))
	    stat_scores.append(np.average(stat_score))

	return percent_diff, run_scores, stat_scores


def get_split_data_decoder(data_set, tuned_cell_dict, mask = False, sub_sample = False, tuning_type = 2, var_thresh = .1):
	all_run_tensors = {}
	all_stat_tensors = {}
	data_tensors = {}

	trialNums = np.array([[i]*50 for i in range(119)])
	trialNums = trialNums.reshape(119*50)

	for key in data_set.keys():
		print key

		#get the fluorescence data, and running speed using wrappers for the AIBS sdk, 
		#optionally filter for tuned cells,

		try:
			dec_cells = tuned_cell_dict[key][tuning_type]  #(0 is for tuned, 1 is for neg, 2 is for pos)
			dff, im_array, stim_table = get_data(data_set[key], stim_info.NATURAL_SCENES)

			cell_inds = data_set[key].get_cell_specimen_indices(dec_cells)
			n_neurons, n_datapoints = dff.shape

			MASK = mask
			SUBSAMPLE = sub_sample

			if MASK:
			    dff = np.ma.array(dff, mask=False)
			    dff.mask[cell_inds] = True
			    dff = dff.compressed()
			    dff.shape = [-1, n_datapoints]
			elif SUBSAMPLE:
			    dff = dff[cell_inds]
			    dff[~np.isfinite(dff)] = 0
			    
			dxcm, dtime = data_set[key].get_running_speed()

			#arrange teh data in a num_cells x num conditions x num_trials x trial length tensor, 
			#then average over trial length

			data_tensor, trialCount = arrange_ns_data_trialTensor(dff, stim_table, offset = 6, lag = 9)  

			ra_variance = get_explainable_variance(data_tensor)

			exp_ind = ra_variance > var_thresh

			dff = dff[exp_ind]	

			data_tensor, trialCount = arrange_ns_data_trialTensor(dff, stim_table, offset = 6, lag = 9)  
			data_tensor = np.average(data_tensor, axis = 3)

			#train_tensor, test_tensor = train_test_split(data_tensor.transpose(2, 0, 1))
			#train_tensor = train_tensor.transpose(1, 2, 0)
			#test_tensor = test_tensor.transpose(1, 2, 0)

			n_neurons, n_conditions, n_trials = data_tensor.shape

			#arrange the running speed in the same tensor, and average the same way
			run_tensor, trialCount = arrange_ns_data_trialTensor(dxcm[np.newaxis, :], stim_table, offset = 6, lag = 9)
			rs = np.average(run_tensor, axis = 3)  

			#linearize so we can index with running speed easily
			rs = rs.reshape([119*50])
			data_tensor = data_tensor.reshape([n_neurons, -1])
			#train_tensor = train_tensor.reshape([n_neurons, -1])
			#test_tensor = test_tensor.reshape([n_neurons, -1])

			data_tensors[key] = (data_tensor.T, rs[:, np.newaxis], trialNums[:, np.newaxis], im_array)

			#want the running trials to be when they are really chugging along. 
			ind_run = rs > 5
			#ind_walk = rs > 1 #np.logical_and(rs > 1, rs <= 15)
			ind_stat = abs(rs) < 2

			subsample = min(sum(ind_run), sum(ind_stat))

			if (subsample < 250):
				pass
			else:

				sub_ind_run = np.random.choice(range(sum(ind_run)), size = subsample) 
				sub_ind_stat = np.random.choice(range(sum(ind_stat)), size = subsample)

				data_run = data_tensor[:, ind_run][:, sub_ind_run]
				#data_run_test = test_tensor[:, ind_run][:, sub_ind_run]

				data_stat = data_tensor[:, ind_stat][:, sub_ind_stat]

				trialNum_run = trialNums[ind_run][sub_ind_run]
				trialNum_stat = trialNums[ind_stat][sub_ind_stat]

				all_run_tensors[key] = (data_run.T, trialNum_run[:, np.newaxis])
				all_stat_tensors[key] = (data_stat.T, trialNum_stat[:, np.newaxis])

				print "the minimum number of running or stationary trials is: ", subsample, " for dataset: ", key
		except:
			print "dataset ", key, " doesn't have tuned neurons"


	print "the number datasets we are returning is:", len(all_run_tensors.keys())
	return all_run_tensors, all_stat_tensors, data_tensors

def calculate_selectivity(tensor):
	n_features = len(tensor)

	trial_av = []
	for i in range(n_features):
	    temp = np.array(tensor[i])
	    n_trials, n_neurons =  temp.shape
	    
	    trial_av.append(np.average(temp, axis = 0))
	    
	trial_av = np.array(trial_av)

	m_res = np.max(trial_av, axis = 0)
	inds = np.argmax(trial_av, axis = 0)   

	mask = np.ones([n_features, n_neurons])
	mask[inds, range(n_neurons)] = 0

	av = np.mean(trial_av[mask.astype('bool')], axis = 0)

	return m_res / ( av + 1e-3)

def calculate_meanvar(tensor):
    
	n_features = len(tensor)

	#first we calculate overall variance

	_, n_neurons = tensor[0].shape
	all_variance = [[] for i in range(n_neurons)]

	for i in range(n_features):
	    num_trials, n_neurons = tensor[i].shape
	    for j in range(n_neurons):
	        for k in range(num_trials):
	            all_variance[j].append(tensor[i][k, j])
	    
	baseline = np.zeros([n_neurons])

	for i in range(n_neurons):
	    baseline[i] = np.var(all_variance[i])
	    
	trial_av = np.zeros([n_features, n_neurons])  + 1e-3

	for i in range(n_features):
	    temp = tensor[i]
	    trial_av[i] = np.mean(temp, axis = 0)
	    
	var_av = np.var(trial_av, axis = 0)
	    
	return var_av, baseline
def calculate_reliability(tensor):
        var_av, baseline = calculate_meanvar(tensor)

        return var_av / baseline
def tensorize(data, labels, shuffle = False, nomean = False, whiten = False, min_class_els = 5):

	labels = labels.flatten()
	n_points, n_neurons = data.shape
	n_classes = len(set(np.array(labels)))

	tensor = [np.empty([0, n_neurons]) for i in range(n_classes)]  
	trial_count = np.zeros(n_classes).astype('int')

	label_map = {sorted(set(np.array(labels)))[i]:i for i in range(n_classes)}

	#print data.shape

	for i in range(n_points):
		if trial_count[label_map[labels[i]]] == 0:
		    tensor[label_map[labels[i]]] = data[np.newaxis, i]
		else:
		    tensor[label_map[labels[i]]] = np.concatenate((tensor[label_map[labels[i]]], data[np.newaxis, i]), axis= 0 )

		trial_count[label_map[labels[i]]] += 1

	num_trials = min(trial_count).astype('int')

	#print num_trials

	if shuffle:

	    for i in range(n_neurons):
	        for j in range(n_classes):
	            tensor[j] = np.array(tensor[j])
	            if len(tensor[j]) > 0:
	                tensor[j][:, i] = np.random.permutation(tensor[j][:, i])


	new_data = np.empty([n_points, n_neurons])
	new_labels = []

	k = 0
	for i in range(n_classes):
	    j = len(tensor[i])
	    if j > min_class_els:
	        for l in range(j):
	            new_labels.append(i)


	        #print tensor[i].shape#, k, k + j, new_data[k:k+j].shape
	        temp = tensor[i]
	        if nomean:
	            mean = np.mean(tensor[i], axis = 0)
	            temp -= mean
	        if whiten:
	            mean = np.mean(tensor[i], axis = 0)
	            for m in range(tensor[i].shape[1]):     
	                temp[:, m] = np.random.normal(mean[m], 0.001, size = temp.shape[0])


	        new_data[k:k+j] = temp
	        k += j

	new_data = new_data[0:len(new_labels)]

	return new_data, np.array(new_labels), tensor

def reliability(all_run_tensors, all_stat_tensors):
    diff_reliability = []
    rel_run = []
    rel_stat = []
    
    for key in sorted(all_run_tensors.keys()):

        data_run, trialNum_run = all_run_tensors[key]
        data_stat, trialNum_stat = all_stat_tensors[key]

        _,_,ten_run = tensorize(data_run, np.array(trialNum_run) - 1)
        _,_,ten_stat = tensorize(data_stat, np.array(trialNum_stat) - 1)

        reliability_run = calculate_reliability(ten_run)
        reliability_stat = calculate_reliability(ten_stat)

        #reliability_run[~np.isfinite(reliability_run)] = 0
        #reliability_stat[~np.isfinite(reliability_stat)] = 0

        diff_reliability.append((2*(np.average(reliability_run) - np.average(reliability_stat)))/ 
                                 (np.average(reliability_run) + np.average(reliability_stat)))
        rel_run.append(np.average(reliability_run))
        rel_stat.append(np.average(reliability_stat))

        #bins = np.linspace(0, 1, num = 20)
        #plt.hist(reliability_run, bins = bins, color = 'c', alpha = .5)
        #plt.hist(reliability_stat, bins = bins, color = 'r', alpha = .5)
        #plt.show()
        #print key
        
    return diff_reliability, reliability_run, reliability_stat

def sparsity(all_run_tensors, all_stat_tensors, POP = True):

#	POP = True

	eps = 1e-3
	s_run_all = []
	s_stat_all = []
	diff_all = []

	for key in all_run_tensors.keys():
	    
		data_run, trialNum_run = all_run_tensors[key]
		data_stat, trialNum_stat = all_stat_tensors[key]

		num_samples, num_neurons = data_run.shape

		trial_av_run = np.zeros([119, num_neurons])  +eps
		trial_av_stat = np.zeros([119, num_neurons]) +eps

		for i in range(119):
			idx, _ = np.where(trialNum_run == i)

			if len(idx) > 0:
				trial_av_run[i] += np.average(data_run[idx], axis = 0)

			idx, _ = np.where(trialNum_stat == i)

			if len(idx) > 0:
				trial_av_stat[i] = np.average(data_stat[idx], axis = 0)
		        

		if POP:
			N = num_neurons
			ax = 1
		else:
			N = 119
			ax = 0
		    
		s_run = (1 - np.sum(trial_av_run, axis = ax)**2 / (N * np.sum(trial_av_run**2, axis = ax)))/(1 - 1./N)
		s_stat = (1- np.sum(trial_av_stat, axis = ax)**2 / (N * np.sum(trial_av_stat**2, axis = ax)))/ (1 - 1./N)

		diff_all.append(2*(np.mean(s_run) - np.mean(s_stat))/ (np.mean(s_run) + np.mean(s_stat)))

		s_run_all.append(np.mean(s_run))
		s_stat_all.append(np.mean(s_stat))

	return diff_all, s_run_all, s_stat_all


def noise_correlations(all_run_tensors, all_stat_tensors):

	n_run_all = []
	n_stat_all = []
	diff_all = []

	for key in all_run_tensors.keys():
		data_run, trialNum_run = all_run_tensors[key]
		data_stat, trialNum_stat = all_stat_tensors[key]

		num_samples, num_neurons = data_run.shape

		corrs_run = []
		corrs_stat = []
		for i in range(num_neurons):
			for j in range(i, num_neurons):
				corrs_run.append(np.correlate(data_run[:, i], data_run[:, j]))
				corrs_stat.append(np.correlate(data_stat[:, i], data_stat[:, j]))

		n_run_all.append(np.mean(corrs_run))
		n_stat_all.append(np.mean(corrs_stat))
		diff_all.append(2*(np.mean(corrs_run) - np.mean(corrs_stat)) / (np.mean(corrs_run) + np.mean(corrs_stat)))

	return diff_all, n_run_all, n_stat_all
