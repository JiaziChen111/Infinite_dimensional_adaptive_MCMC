function S = materncorr(dist,kappa,nu)
%%
%	Creating a Matern correlation function
%
%
%%
S = 2^(1-nu)/gamma(nu)*((kappa*dist).^nu).*besselk(nu,kappa*dist);
S(find(speye(size(dist,1)))) = 1;
S(isnan(S)) = 1;
