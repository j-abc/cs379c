%% now i want to generate pretty pictures for each of these


% iterate across cells

% for all my cells...
% create a running image list

%%
% work to do for this thing
% and how to do that work
% what is best??

% what to do and how to do it...

%%
% save a bunch of matlab figures into one file
path_list = ab_list_paths;
exp_df = readtable(path_list.meta);
iexp = 4;
my_container_id = exp_df{iexp,'experiment_container_id'};
num_cells= exp_df{iexp,'num_cells'};
for icell = 0:num_cells-1
    clf;
    % build file paths
    my_params = {'cell_rel_idx',num2str(icell);
        'experiment_container_id',num2str(my_container_id)};
    my_paths = util_flood_list(my_params, path_list);
    
    % load the data structure
    rf = load(my_paths.rf);
    f = figure(1);
    set(f, 'Position',[20 20 1049 300]);
    my_fields = {'allen','ridge','ald'};
    for ifield = 1:length(my_fields)
        my_field = my_fields(ifield);
        try
        subplot(1,3,ifield);
        imagesc(rf.(my_field{:}));
        colorbar;
%         caxis([-0.001 0.01]);
        catch
        end
    end
    pause;
end
%%
% examine ALDf, s, and sf
f = figure(2); 
set(f, 'Position',[20 20 1049 300]);
subplot(1,3,1); imagesc(reshape(rf.ald_store.khatF,[16 28])); hold on;
subplot(1,3,2); imagesc(reshape(rf.ald_store.khatS,[16 28])); hold on;
subplot(1,3,3); imagesc(reshape(rf.ald_store.khatSF,[16 28])); hold on;

% questions:
%   does scale matter?
%   how does scale matter if it does?
%   what are we supposed to do wiht the scaling?
%   am i doing the receptive fields properly?