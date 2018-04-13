function model = GPCreate(type, loc, param)
%%
%   creates a Gaussian processes object,
%   for tests
%
%   type  - Matern
%
%   loc  - location of points (n x d)
%      
%   param  - parameters
%            for Matern
%                  1 - variance
%                  2 - length-scale (kappa)
%                  3 - nu           (diffrentiability)
%           for exp (I know speical case of matern bla bla)
%                  1 - variance
%                  2 - length scale
%           gauss (Stein would not be happy)
%                  1 - variance
%                  2 - length scale  
%           SPDE-Materin
%                  1 - variance
%                  2 - kappa                 
%%

model = struct;
if strcmp(type,'Matern')
    model.type = type;
    model.loc = loc;
    dist = squareform(pdist(loc));
    model.C = param(1) * materncorr(dist,param(2),param(3));
elseif strcmp(type,'exp')
    model.type = type;
    model.loc = loc;
    dist = squareform(pdist(loc));
    model.C = param(1) * exp(-dist/param(2));
    
elseif strcmp(type,'gauss')    
    model.type = type;
    model.loc = loc;
    dist = squareform(pdist(loc));
    model.C = param(1) * exp(-dist.^2/param(2));
    jitter = 1e-8;
    model.C = model.C + jitter * eye(size(model.C));
elseif strcmp(type,'SPDE')       
    [A,P,FV,C,G]=create_triangulation(loc,0,[0.08 0.08],[0.05 -0.2],20);
    model.type = type;
    model.loc = loc;

    model.P = P;
    model.A = A;
    sigma = param(1);
    kappa = param(2);
    K1 = (G + kappa^2*C);
    alpha = 2;
    c = sqrt(gamma(alpha-1)./gamma(alpha))./(kappa.^(alpha-1).*(4*pi)^(1/2).*sigma);
    Ci = sparse(diag(1./sum(C)));
    model.Q = c ^2 * (K1*Ci*K1);
    model.reo  = symamd(model.Q +  A'* A);
    model.Q = model.Q(model.reo,model.reo);
    
    model.C = full(inv(model.Q));
    model.A = A(:,model.reo);
    model.FV = FV;
end

[U, E]=eig(model.C);
[~ ,permutation]=sort(diag(E),'descend');
E = E(permutation,permutation);
U = U(:,permutation);
model.U = U;
model.Lambda = diag(E);
model.invLambda = 1./model.Lambda;
model.X = zeros(size(model.C,1), 1);
model.n = length(model.X);
%model.C_chol = chol(model.C)';
%model.Cinv_chol = chol(inv(model.C))';
