function Nh = mcmc_ess(x)
%%
%
% stolen from Titsias
%%
x = x(:);

N = length(x);
g = autocorr(x,N-1);



d = find((g)<0.05, 1);
if isempty(d)
  d = length(g);
end

%
v = 1/N * (g(1) + 2*sum(g(2:(d-1))));
Nh = g(1) / v;
%[~,d] = find(


% error('not implemented')