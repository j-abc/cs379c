function path_list = ab_list_paths
%% define paths of interest (for data)
path_cell = {'data','[base]/data/','parameter';
             'mdff','[base]/data/mdff/[experiment_container_id]/mdff_[experiment_container_id]_[cell_rel_idx].mat','parameter';
             'img','[base]/data/img/img_[experiment_container_id].mat','parameter';
             'meta','[base]/data/meta.csv','parameter';
             'rf','[base]/data/rf/[experiment_container_id]/rf_[experiment_container_id]_[cell_rel_idx].mat','parameter';
             'res_rf','[base]/data/res_rf/[experiment_container_id]/rf_[experiment_container_id]_[cell_rel_idx].png','parameter'} ;
path_list = util_extract_inputs(path_cell,{});
path_list = util_flood_list({'base','.'}, path_list);
return
