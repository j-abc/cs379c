function out_struct = ab_aldsf_format(prs)
% Unpack params
nsevar = abs(prs(1)); % to avoid nsevar<0
prs = prs(2:end); % Hyper-params, except noise variance

nt = 1;
nx = 16*28;
leng_ndims = 2;

% ALDs
numbTheta_s = length(prs(1:5)); % # of parameters in ALDs prior cov. mat.

%% ALDs
v_mu = prs(1:2); % unpack params

% precision/covariance matrix
v_gam = prs(3:4);
v_phi = prs(5);

inds = find(1-tril(ones(leng_ndims))); % make precision matrix L
Lmult = ones(leng_ndims);
Lmult(inds) = v_phi;
Lmult = Lmult.*Lmult';
L = v_gam*v_gam';
L = L.*Lmult; % we want the inverse of this thing to get our covariance matrix

%% ALDf
numbTheta_f = length(prs(6:10)); % # of parameters in ALDf prior cov. mat.

M = [prs(6) prs(7); prs(7) prs(8)]; % unpack params
vmu = [prs(9); prs(10)];

%%
[ft, fx] =  FFT2axis(nt,nx); % coordinates in Fourier domain
w = [ft(:) fx(:)];

muvec = repmat(vmu, 1, length(w));
absMw = abs(M*w');
W2 = (absMw - muvec)'; % move toward the mean
vF = exp(-.5*sum(W2.*W2,2)); % diagonal of ALDf prior cov. matrix.

sign_absMw = sign(M*w')'; % for computing derivatives

nrmtrm = sqrt(nt*nx);  % normalization term to make FFT orthogonal
BB = FFT2matrix(nt,nx)/nrmtrm;

ovsc_sf = prs(end);
%%
out_struct.corr = inv(L);
out_struct.det  = det(out_struct.corr);

ev = eig(inv(L));
jim = abs(ev);
maxi = max(jim);
mini = min(jim);
out_struct.ecc = sqrt(1-mini/maxi);

out_struct.mu_space = v_mu;
out_struct.mu_freq = vmu;
%%
return