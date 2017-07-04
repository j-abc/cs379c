function data_struct = ab_load_data(my_paths)
load_list = {'mdff','img'};
data_struct = struct();
for iload = load_list
    temp = load(my_paths.(iload{:}));
    data_struct = setstructfields(data_struct, temp);
end
data_struct.img = double(data_struct.img/255);

data_struct.nkt = 1;

[data_struct.n_samp, n_y, n_x] = size(data_struct.img);
data_struct.img_dims = [n_y, n_x];

data_struct.flat_img = reshape(data_struct.img,[data_struct.n_samp prod(data_struct.img_dims)]);
data_struct.mdff  = permute(double(data_struct.mdff), [2 1]);
return