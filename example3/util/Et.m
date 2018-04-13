
% from Z to U
%   
%   Z - [N, M]
%%
function [Z] = Et(sigma, alpha, tau, U, K1, K2)
	

    [N, M]  = size(U);
    U = reshape(dct2(U'), N, M);
    
	if nargin < 5
        [K1,K2] = meshgrid(0:N-1,0:M-1);
    end
    c = sigma * tau * sqrt(4 * pi * gamma(1.5)/gamma(0.5));
    
    coeff = (pi^2*(K1.^2+K2.^2) + tau^2).^(-alpha/2);
    coeff = c*sqrt(N*M)*coeff(:);

	% Construct the KL coefficients
	coeff(1) = 0;
	Z = coeff.*U(:);
    Z = reshape(Z, N, M);
    
end