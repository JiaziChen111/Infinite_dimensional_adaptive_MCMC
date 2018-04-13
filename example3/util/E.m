
% from Z to U
%   
%   Z - [N, M]
%%
function [U, coeff] = E(sigma, alpha,tau, Z , K1, K2)
	
	% Random variables in KL expansion
% 	xi = normrnd(0,1,N);
%     if nargin > 3
%         xi(:) = D .* xi(:);
%     end

 	[N, M]  = size(Z);
    if nargin < 5
	% Define the (square root of) eigenvalues of the covariance operator
        [K1,K2] = meshgrid(0:N-1,0:M-1);
    end

    c = sigma * tau * sqrt(4 * pi * gamma(1.5)/gamma(0.5));
    coeff = (pi^2*(K1.^2+K2.^2) + tau^2).^(-alpha/2);
    coeff = c*sqrt(N*M)*coeff(:);

	% Construct the KL coefficients
	U = coeff.*Z(:);
    U(1,1) = 0;
	
    U = idct2(reshape(U, M, N))';
    
end