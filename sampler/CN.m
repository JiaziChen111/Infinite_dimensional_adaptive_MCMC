function [model, samples, accRates] = CN_centered_scaled_adaptive(model, mcmcoptions)
%%
%function [model samples accRates] = gpsamppCN_fixedhypers(model, mcmcoptions)
%
% pCP centering around the posterior mean for latent Gaussian models  
%
% Inputs: 
%         -- model: the structure that contains the log likelihood and the latent
%                    Gaussian prior parameters such as the covariance matrix 
%                    with its pre-computed eigendecomposition 
%         -- mcmcoptions: user defined options about the burn-in and sampling iterations
%                      and others (see demos)
%
% Outputs: model: 
%         -- model: as above. The outputed model is updated to contain the
%                   state vector of the final MCMC iteration
%                   as well as the learned step size delta
%         -- samples: the structure that contrains the samples 
%         -- accRates: acceptance rate
%
%
% orig: Michalis K. Titsias (2016)
% mod: Jonas Wallin (2017)
%%

BurnInIters = mcmcoptions.Burnin; 
Iters = mcmcoptions.T; 
StoreEvery = mcmcoptions.StoreEvery;
n = length(model.X);
num_stored = floor(Iters/StoreEvery);
X = model.X;
samples.X = zeros(num_stored, n);
samples.LogL = zeros(1, num_stored);
samples.num_calls = 0;



samples.LogL = zeros(1, BurnInIters + Iters);
samples.ff  = zeros(1,  BurnInIters + Iters);

% compute the initial values of the likelihood p(Y | F)

oldLogLik = model.likelihood(model, X);

model.delta = 2/sqrt(n);
model.beta = 2/(model.delta + 2);
beta = model.beta;

cnt = 0;

range = 0.05;
%setting the accpetance rate
if isfield(model,'opt')
   opt = model.opt;
   range = 0; 
else
    opt = 0.2;
end


% adaption step size
epsilon = 0.2;

acceptHistF = zeros(1, BurnInIters + Iters);
count_m = 0;
E  = model.U' * ( sqrt(model.invLambda).* X);
% sample using Neal's method

n = length(model.E_mean);
model.A = model.A* model.U*diag(sqrt(model.Lambda));
model.U = 1;
for it = 1:(BurnInIters + Iters) 
%     
    if(mod(it, 10000)==0)
        fprintf('iter = %d (%.2f) (beta = %.2f)\n',it,mean(acceptHistF(1:it-1)), beta);
     end
    %sampling step
    sqrtOneBeta = sqrt(1-beta^2);
    Enew =    sqrtOneBeta.*E + beta.*randn( model.n, 1);
    %
    %Xnew=   model.U * (sqrt(model.Lambda).* Enew);
    Xnew = Enew;
    newLogLik = model.likelihood(model, Enew);
     
    % Metropolis-Hastings to accept-reject the proposal
     hxy = newLogLik;% 
     hyx = oldLogLik;
     
    [accept, uprob] = metropolisHastings(hxy, hyx, 0, 0); 
   
    acceptHistF(it) = accept;  
     
    if accept == 1
        X = Xnew;
        E = Enew;   
        oldLogLik = newLogLik; 
    end
    
    
    
    % Adapt proposal during burnin
    if mod(it,5) == 0 
        
       
        
        if (it >= 50) 
            accRateF = mean(acceptHistF((it-49):it))*100;
                if (accRateF > (100*(opt+range))) || (accRateF < (100*(opt-range)))
                
                    count_m = count_m + 1;
                    epsilon_n = sqrt(epsilon^2 /count_m); 
                    model.delta = model.delta + (epsilon_n*((accRateF/100 - opt)/opt));
                    model.beta = exp(model.delta)/(1+ exp(model.delta));
                    beta = model.beta;
                end
        end
    end
    
    % keep samples after burn in
    if (it > BurnInIters)  & (mod(it,StoreEvery) == 0)
    %
        cnt = cnt + 1;
        samples.X(cnt,:) = X;  
    %
    end
    %    
    samples.LogL(it) = oldLogLik;
    samples.ff(it) = 0.5*(X'*X);    
end
%
%
model.X = X;
accRates.X = mean(acceptHistF(BurnInIters+1:end))*100; 

 

 
