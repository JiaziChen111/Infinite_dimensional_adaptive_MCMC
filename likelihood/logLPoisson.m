function lik= logLPoisson(model, X, grad)
%function out = logLPoisson(model, Y, X)
%
%Description:  Poisson likelihood:
%              latent parameters are
%              model.offstet - offstet parameter
%              model.v       - mean
%

if nargin < 3
	grad  = 0;
end

Xv = model.A * X + model.v;
if grad==0
	lik = sum(model.Y .* Xv - model.offset .* exp(Xv));
else
	lik = model.A'*(model.Y - model.offset .* exp(Xv));
end

