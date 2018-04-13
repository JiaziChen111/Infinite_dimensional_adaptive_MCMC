function [lik, grad, H] = logLBinom(model, X)
%function out = logLBinom(model, Y, X)
%
%Description:  binom-loigt likelihood:
%              latent parameters are
%              model.mu       - mean
%              model.nY       - number of observation 
%


Xmu = model.A * X + model.mu;

eX = exp(Xmu);
e1X = (1+eX);
p  = eX./e1X;
lik = sum(model.Y .* log(p) + (model.nY - model.Y) .* log(1-p));
if nargout > 1
	grad = model.A' * (model.Y./e1X - ...
                      (model.nY - model.Y).*p ) ; 
end
if nargout > 2
    n_Y = length(model.Y);
    H  = model.A'*sparse(1:n_Y,1:n_Y, model.nY.*(p.^2./eX) ) * model.A; % - Hessian
end

