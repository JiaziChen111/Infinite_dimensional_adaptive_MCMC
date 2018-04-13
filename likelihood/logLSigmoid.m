function [lik, grad, H] = logLSigmoid(model, X, grad)
%function out = logLSigmoid(model, X, grad)
%
%Description:  log-likelihood for binary classification
%

AX = model.A * X;
YX = model.Y.* AX; 
lik = sum(-log(1 + exp(-YX)));
if nargout > 1
	grad = model.A'*( 0.5* (model.Y+1) - 1./(1+exp(-AX)));
end
if nargout > 2
    fprintf('not implminted\n');
    H  = 1000;
end
