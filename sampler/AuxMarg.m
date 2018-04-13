function [model samples accRates] = AuxMarg(model, mcmcoptions)
%function [model samples accRates] = gpsampAuxMarg_fixedhypers(model, mcmcoptions)
%
% The marginal gradient-based method for latent Gaussian models 
%
% Inputs: 
%         -- model: the structure that contains the log likelihood and the latent
%                    Gaussian prior parameters such as the covariance matrix 
%                    with its pre-computed eigendecomposition (see demos) 
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
% Michalis K. Titsias (2016), J Wallin (2018)
%


BurnInIters = mcmcoptions.Burnin; 
Iters = mcmcoptions.T; 
StoreEvery = mcmcoptions.StoreEvery;
n = length(model.X);
num_stored = floor(Iters/StoreEvery);
X = model.X';
samples.X = zeros(num_stored, n);
samples.LogL = zeros(1, num_stored);
samples.num_calls = 0;



samples.LogL = zeros(1, BurnInIters + Iters);
samples.ff  = zeros(1,  BurnInIters + Iters);



model.delta = 2/sqrt(n); 
twoLambda = 2*model.Lambda;
fourLambda = 4*model.Lambda;

% Quantities that need to be updated when the state vector
% the step size delta change (only during adaption)
deltaLambda = model.delta*model.Lambda;
Lambda1 = deltaLambda./(twoLambda + model.delta);
Lambda3 = (twoLambda + model.delta)./(fourLambda + model.delta);
sqrtLambda2 = sqrt( deltaLambda.*(fourLambda + model.delta) )./(twoLambda + model.delta);

% Quantities that need to be updated when the state vector
% changes
[oldLogLik,derF] = model.likelihood(model, X');  
FU = X*model.U;
derFU = model.U'*derF;   
partOfMeanSamp = ( (2/model.delta)*FU  + derFU').*(Lambda1');
partOfMeanMH = ( (2/model.delta)*FU  + 0.5*derFU').*(Lambda1');
Lambda3derFU = Lambda3.*derFU;

cnt = 0;

range = 0.05;
opt = 0.54;

% adaption step size
epsilon = 0.05;

acceptHistF = zeros(1, BurnInIters + Iters);

for it = 1:(BurnInIters + Iters) 
%
    if(mod(it, 10000)==0)
        beta = sqrt(1- (2 / (2 + model.delta))^2) ; 
        fprintf('iter = %d (%.2f) (beta = %.2f)\n',it,mean(acceptHistF(1:it-1)),beta);
        
     end
    % Propose new state vector F
    Xnew = (randn(1, n).*(sqrtLambda2') + partOfMeanSamp)*model.U';
            
    % perform an evaluation of the likelihood p(Y | F) 
    % new gradient 
    [newLogLik, derFnew] = model.likelihood(model, Xnew');     
    FUnew = Xnew*model.U;
    derFUnew = model.U'*derFnew;   
    partOfMeanSampnew = ( (2/model.delta)*FUnew + derFUnew').*(Lambda1');
    partOfMeanMHnew = ( (2/model.delta)*FUnew + 0.5*derFUnew').*(Lambda1');
    Lambda3derFUnew = Lambda3.*derFUnew;
         
    hxy = (FU -  partOfMeanMHnew)*Lambda3derFUnew;
    hyx = (FUnew -  partOfMeanMH)*Lambda3derFU;
    corrFactor = hxy - hyx; 
    
    [accept, uprob] = metropolisHastings(newLogLik+corrFactor, oldLogLik, 0, 0); 
    
    acceptHistF(it) = accept;   
      
    if accept == 1
         X = Xnew;
         FU = FUnew;
         derFU = derFUnew;   
         partOfMeanSamp = partOfMeanSampnew;
         partOfMeanMH = partOfMeanMHnew;
         Lambda3derFU = Lambda3derFUnew;
         oldLogLik = newLogLik; 
    end
    
    % Adapt proposal during burnin 
    if mod(it,5) == 0 
        if (it >= 50) 
        accRateF = mean(acceptHistF((it-49):it))*100;
        if (it <= BurnInIters)
        if (accRateF > (100*(opt+range))) || (accRateF < (100*(opt-range)))
        %    
            model.delta = model.delta + (epsilon*((accRateF/100 - opt)/opt))*model.delta;
   
            deltaLambda = model.delta*model.Lambda;
            Lambda1 = deltaLambda./(twoLambda + model.delta);
            Lambda3 = (twoLambda + model.delta)./(fourLambda + model.delta);
            sqrtLambda2 = sqrt( deltaLambda.*(fourLambda + model.delta) )./(twoLambda + model.delta);
            partOfMeanSamp = ( (2/model.delta)*FU + derFU').*(Lambda1');
            partOfMeanMH = ( (2/model.delta)*FU + 0.5*derFU').*(Lambda1');
        %    
        end
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
    
    samples.LogL(it) = oldLogLik;
    samples.ff(it) = 0.5*(X*X');
    %        
end
%
%
model.beta = beta;
model.F = X;
accRates.X = mean(acceptHistF(BurnInIters+1:end))*100; 

