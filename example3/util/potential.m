function [lik, grad] = potential(Y, A, U, meanY, cellArea)
%%
%   likelhiood and gradient of the likelihood
%
%
%
%%
UY  = A*U + meanY;
EUY =  exp(UY);
lik = sum(Y .* UY - cellArea * EUY);

if nargout > 1
    
    grad = A'*(Y - cellArea * EUY);
end

