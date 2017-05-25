function data_struct = load_data(my_paths)
load_list = {'mdff','y'};
data_struct = struct();
for iload = load_list
    temp = load(my_paths.(iload{:}));
    data_struct = setstructfields(data_struct, temp);
end
return