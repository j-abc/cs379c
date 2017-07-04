%%
clear; clc;

path_list = ab_list_paths;
meta_df = readtable(path_list.meta);

sel_subj = 1:16;
sel_df = meta_df(sel_subj,:);

irow = 1;
icell = 1;

cntr = 1;
for irow = sel_subj
    params = table2struct(sel_df(irow,:));
    for icell = 0:(params.num_cells-1)
        params.cell_rel_idx = icell;
        rf_file = util_fill_params(params,path_list.rf);
        % gather metrics of interest
        try
            rf = load(rf_file);
            out_struct = ab_gather_metrics(rf.ald_store.thetaSF);
            %             out_struct.evidSF = rf.ald_store.evidSF; % not sure how to
            %             use this, really...
            out_struct.rf = rf.ald;
            out_struct = catstruct(out_struct, params);
            all_struct(cntr,1) = out_struct;
            cntr = cntr + 1;
        catch
        end
    end
    fprintf('subj %0.0f done\n',irow);
end

out_table = struct2table(all_struct);
save('./data/results.mat','out_table')

%%

%%

%%

clear; clc;

path_list = ab_list_paths;
meta_df = readtable(path_list.meta);

sel_subj = 1:16;
sel_df = meta_df(sel_subj,:);

irow = 1;
icell = 1;

cntr = 1;
for irow = sel_subj
    params = table2struct(sel_df(irow,:));
    for icell = 0:5:(params.num_cells-1)
        tic
        params.cell_rel_idx = icell;
        % save out
        img_file = util_fill_params(params,path_list.res_rf);
        [img_dir, ~] = fileparts(img_file);
        
        if ~exist(img_file)
            rf_file = util_fill_params(params,path_list.rf);
            try
                rf = load(rf_file);
                % build figure
                f = figure(1);
                set(f, 'Position',[20 500 1600 500]);
                my_fields = {'allen','ridge','ald'};%,'khatS','khatF'};
                for ifield = 1:length(my_fields)
                    my_field = my_fields(ifield);
                    try
                        subplot(1, 3, ifield);
                        if ismember(my_field{:},{'allen','ridge','ald'})
                            imagesc(rf.(my_field{:})');
                        else
                            imagesc(reshape(rf.ald_store.(my_field{:}),[16, 28]));
                        end
                        colorbar;
                        %                 title(my_field{:});
                        pbaspect([16*5 28*5 1]);
                        axis off;
                    catch
                    end
                end
            catch
            end
            
            if ~isdir(img_dir)
                mkdir(img_dir);
            end
            saveas(f, img_file);
            fprintf('%0.0f, %0.0f, %0.2fs\n', irow, icell, toc);
        end
    end
end
%%