function [rf, fail] = ab_get_rf(ds)
rf = struct();
rf.allen = ab_run_allen_rf(ds.mdff, ds.img);
try
[rf.ald_store, rf.ridge] = runALD(ds.flat_img, ds.mdff, [ds.img_dims(1); ds.img_dims(2)], 1);
rf.ald = reshape(rf.ald_store.khatSF, ds.img_dims(1), ds.img_dims(2));
rf.ridge = reshape(rf.ridge, ds.img_dims(1), ds.img_dims(2));
fail = 0;
catch
    rf.ald = struct();
    rf.ridge = [];
    fail = 1;
end
return