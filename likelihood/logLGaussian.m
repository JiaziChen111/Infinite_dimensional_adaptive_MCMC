function lik= logLGaussian(model, X, grad)
%function out = logLGaussian(model, Y, X)
%
%Description: simple Gaussian error observations,
% if grad then compute the gradient w.r to X
%

if nargin < 3
	grad  = 0;
end

if grad==0
	lik = - (0.5/model.sigma.^2) * (model.Y - model.A * X)'*(model.Y - model.A * X);
else
	lik = (1/model.sigma.^2) * model.A'*(model.Y - model.A * X);
end

