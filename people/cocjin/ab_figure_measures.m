figure(1);
out_table = sortrows(out_table,'det');
my_idx = fliplr([7 1008 1505 1750 2112]);
for i = 1:5
    subplot(1,5,i)
    imagesc(out_table{my_idx(i),'rf'}{:}');
    pbaspect([16*5 28*5 1]);
    axis off;
end

%%
figure(2);
out_table = sortrows(out_table,'ecc');
% my_idx = round(linspace(3,height(out_table),6));
my_idx = [35 333 1500 1950 2112];
for i = 1:5
    subplot(1,5,i)
    imagesc(out_table{my_idx(i),'rf'}{:}');
    pbaspect([16*5 28*5 1]);
    axis off;
end

%%
dmean= @(x) mean(abs(x));
figure(2);
out_table.mf = cellfun(dmean, out_table.mu_freq);
out_table = sortrows(out_table,'mf');
% my_idx = [5 100 1000 2000 2100 2115];
% my_idx = [10 100 500 2115 2110];
my_idx = 2:10;
for i = 1:5
    subplot(1,5,i)
    imagesc(out_table{my_idx(i),'rf'}{:}');
    pbaspect([16*5 28*5 1]);
    axis off;
end
%%
dmean= @(x) mean(abs(x));
figure(2);
out_table.mf = cellfun(dmean, out_table.mu_freq);
out_table = sortrows(out_table,'mf');
% my_idx = [5 100 1000 2000 2100 2115];
% my_idx = [10 100 500 2115 2110];
my_idx = 2:5:100;
for i = 1:5
    subplot(1,5,i)
    imagesc(out_table{my_idx(i),'rf'}{:}');
    pbaspect([16*5 28*5 1]);
    axis off;
end

%%
dmean= @(x) mean(abs(x));
figure(2);
out_table.mf = cellfun(dmean, out_table.mu_freq);
out_table = sortrows(out_table,'mf');
% my_idx = [5 100 1000 2000 2100 2115];
my_idx = [2050:9:2100];
for i = 1:5
    subplot(1,5,i)
    imagesc(out_table{my_idx(i),'rf'}{:}');
    pbaspect([16*5 28*5 1]);
    axis off;
end
%%
dmean= @(x) mean(abs(x));
figure(2);
mu_img = [8 14]';
out_table.d2c = cellfun(@(x) sqrt(sum((mu_img-x).^2)), out_table.mu_space);
out_table = sortrows(out_table,'d2c');
% my_idx = [5 100 1000 2000 2100 2115];
my_idx = [10 300 750 1000 2000];
for i = 1:5
    subplot(1,6,i)
    imagesc(out_table{my_idx(i),'rf'}{:}');
    pbaspect([16*5 28*5 1]);
    axis off;
end

%%