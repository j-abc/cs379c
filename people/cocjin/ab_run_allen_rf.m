function mean_img = ab_run_allen_rf(mdff, img)
%%
fdoty = bsxfun(@times, mdff, img);
mean_img = squeeze(mean(fdoty,1));
return