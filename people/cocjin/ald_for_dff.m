%% define paths of interest (for data)
path_cell = {'data','[base]/data/','parameter';
             'mdff','[base]/data/mdff/mdff_[experiment_container_id]_[cell_rel_idx].mat','parameter';
             'y','[base]/data/y/y_[experiment_container_id].mat','parameter';
             'meta','[base]/data/meta.csv','parameter'} ;
path_list = util_extract_inputs(path_cell,{});
path_list = util_flood_list({'base','.'}, path_list);
%% read in the per-container metadata table
meta_df = readtable(path_list.meta);

%% convert this to a for loops for actually loading/executing data
% we will iterate across containers with ic
ic = 1;

% indices for accessing files
my_cell_idx = 0;
my_cont_id = meta_df{ic,'experiment_container_id'};

% build file paths
my_params = {'cell_rel_idx',num2str(my_cell_idx);
    'experiment_container_id',num2str(my_cont_id)};
my_paths = util_flood_list(my_params, path_list);

%% this is what goes in the above for loop, for loading
% load our data into a structure
data_struct = ab_load_data(my_paths);

%% reshape data
nkt = 1;
[n_samp,n_x, n_y]= size(data_struct.y); % filterdims = [nx ny], nkt = 1
data_struct.y_flat = reshape(double(data_struct.y),[n_samp n_x*n_y]);
data_struct.x_in = permute(double(data_struct.mdff), [2 1]);
%% this is what goes in the above for loop, for loading
% Stimuli[2500x324]
% y[samp x 1]
[khatALD, kridge] = runALD(data_struct.y_flat, data_struct.x_in, [n_y; n_x], nkt);

%%

% change y to x
% now that we have the data, we can try out jonathan's code