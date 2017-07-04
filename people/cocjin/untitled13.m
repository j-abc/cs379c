clear; clc; close all;

%% read in the per-container metadata table
path_list = ab_list_paths;
exp_df = readtable(path_list.meta);  % experiment dataframe

%%
end_iter = 16;% height(exp_df);
for iexp = 1:end_iter
    % get the id for this container
    my_container_id = exp_df{iexp,'experiment_container_id'};
    
    % loop across cells
    num_cells = exp_df{iexp, 'num_cells'};
    fprintf('starting: %0.0f experiment out of %0.0f\n',iexp, end_iter);
    fprintf('starting: %0.0f experiment out of %0.0f\n',iexp, end_iter);
    fprintf('starting: %0.0f experiment out of %0.0f\n',iexp, end_iter);
    fprintf('starting: %0.0f experiment out of %0.0f\n',iexp, end_iter);
    fprintf('starting: %0.0f experiment out of %0.0f\n',iexp, end_iter);
    for icell = 0:(num_cells-1)
        % build file paths
        my_params = {'cell_rel_idx',num2str(icell);
            'experiment_container_id',num2str(my_container_id)};
        my_paths = util_flood_list(my_params, path_list);
        
        % load the data structure
        ds = ab_load_data(my_paths);

        % get our rf structure
        [pathstr, name, ext] = fileparts(my_paths.rf);
        if ~exist(pathstr)
            mkdir(pathstr)
        end
        if ~exist(my_paths.rf)
            [rf, fail] = ab_get_rf(ds);
            save(my_paths.rf, '-struct','rf');
            if fail
                fprintf('fail %0.0f\n',icell)
            end
        else
            fprintf('%0.0f already run!\n',icell)
        end
    end
    fprintf('ending: %0.0f experiment out of %0.0f\n',iexp, end_iter);
    fprintf('ending: %0.0f experiment out of %0.0f\n',iexp, end_iter);
    fprintf('ending: %0.0f experiment out of %0.0f\n',iexp, end_iter);
    fprintf('ending: %0.0f experiment out of %0.0f\n',iexp, end_iter);
    fprintf('ending: %0.0f experiment out of %0.0f\n',iexp, end_iter);
    fprintf('ending: %0.0f experiment out of %0.0f\n',iexp, end_iter);
end
1%%
rf = load(my_paths.rf);
figure(1); imagesc(rf.ald)
figure(2); imagesc(rf.allen)
figure(3); imagesc(rf.ridge)

%% convert this to a for loop for actually loading/executing data
% we will iterate across containers with ic
ic = 1;

% indices for accessing files
my_cell_idx = 100;
my_cont_id = meta_df{ic,'experiment_container_id'};

% build file paths
my_params = {'cell_rel_idx',num2str(my_cell_idx);
    'experiment_container_id',num2str(my_cont_id)};
my_paths = util_flood_list(my_params, path_list);

%% this is what goes in the above for loop, for loading
% load our data into a structure

%% reshape data


%% this is what goes in the above for loop, for loading
% Stimuli[2500x324]
% y[samp x 1]
% input 1: xtraining: 2D
% input 2: 1D
[khatALD, kridge] = runALD(data_struct.flat_in, data_struct.dff_in, [data_struct.n_y; data_struct.n_x], nkt);
 
rf.ald    = 
rf.ridge  = 
rf.allen  =
%%
figure(1); imagesc(reshape(kridge, n_x, n_y))
figure(2); imagesc(reshape(khatALD.khatSF, n_x, n_y));
%%
datastruct = formDataStruct(x, y, nkt, spatialdims);

numb_dims = length(datastruct.ndims);
opts0.maxiter = 1000;  % max number of iterations
opts0.tol = 1e-6;  % stopping tolerance
lam0 = 10;  % Initial ratio of nsevar to prior var (ie, nsevar*alpha)
% ovsc: overall scale, nasevar: noise variance
[kRidge, ovsc ,nsevar]  =  runRidge(lam0, datastruct, opts0);
%%
fdoty = bsxfun(@times, double(data_struct.mdff)', double(data_struct.y));
mean_img = squeeze(mean(fdoty,1));
figure(3); imagesc(mean_img);
%%

%%
% to do:
%   1] check for robusteness of our receptive field maps under the presence of noise
%   2] how to check for robustness...
%      inject noise into the algorithm
%      and test if it's reasonable or not. 
%      do we have a metric on this??

% change y to x
% now that we have the data, we can try out jonathan's code

%%
% nsevar gave wrong output, runALD. nsevar from runRidge.
% runRidge gave all zeros (why? this isn't expected, right?)


%%
% function [khatALD, kRidge] = runALD(x, y, spatialdims, nkt)

% 

% compare runRidge inputs/outputs between
%   synthetic and real stimuli
%   check the shapes of all of these variables.
%   change the input dimensions of the synthetic stimuli
%   28 x 16

% extract
% could be ;interesting:
%   extract out the allen atlas receptive fields
%   create synthetic data from that
%   and then see if we can get it back.
%   
%%
% http://help.brain-map.org/display/observatory/Documentation?preview=/10616846/10813485/VisualCoding_VisualStimuli.pdf
% herpadper how to do this stuff... 

% how to best characterize these responses...
