% compare these...
% err
% dr
% det
% ecc
out_table.freq_x = cellfun(@(x) x(1), out_table.mu_freq);
out_table.freq_y = cellfun(@(x) x(2), out_table.mu_freq);
uni_struc = unique(out_table.targeted_structure)';
%%
cnt = 1;
compare_fields = {'det','ecc','freq_x','freq_y'};%,'mf'};
for ifield = compare_fields
    g(1,cnt) = gramm('color',out_table.targeted_structure,'x',out_table.(ifield{:}));
    g(1,cnt).stat_density();%('function','normalization');%'fill','transparent');
    switch ifield{:};
        case 'ecc'
            name = 'Eccentricity';
        case 'mf'
            name = 'Mean spatial frequency';
        case 'det'
            name = 'Determinant';
        case 'freq_x'
            name = 'Mean frequency in x';
        case 'freq_y'
            name = 'Mean frequency in y';
    end
    g(1, cnt).set_names('x',name)
    cnt = cnt + 1;
end
g.draw()
%%


[G, strucs] = findgroups(out_table.targeted_structure);
%%
res.mean = splitapply(@mean, out_table.det, G);
res.std  = splitapply(@std, out_table.det, G);
res.max  = splitapply(@max, out_table.det, G);
res.min  = splitapply(@min, out_table.det, G);
res.strucs = strucs;
struct2table(res)

%%
fre.mean = splitapply(@mean, [out_table.mu_freq{:}]',G)
fre.std  = splitapply(@std, [out_table.mu_freq{:}]',G)
fre.max  = splitapply(@max, [out_table.mu_freq{:}]',G)
fre.min  = splitapply(@min, [out_table.mu_freq{:}]',G)
fre.strucs = strucs
struct2table(fre)

%%
