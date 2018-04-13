function [model, samples, accRates] = LCN_hmeasure(model, mcmcoptions)
%%
%   Langevian Crank Nichlson with changed base measure, 
%   The measure uses the Hessian of the likelihood
%   
%
% Inputs: 
%         -- model: the structure that contains the log likelihood and the latent
%                    Gaussian prior parameters such as the covariance matrix 
%                    with its pre-computed eigendecomposition 
%         -- mcmcoptions: user defined options about the burn-in and sampling iterations
%                      and others 
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
% mod: Jonas Wallin (2087)
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
    opt = 0.5;
end


% adaption step size
epsilon = 0.2;
count_m=1;

acceptHistF = zeros(1, BurnInIters + Iters);

if strcmp(model.adaptive,'eigen')
    E  = model.U' * ( sqrt(model.invLambda).* X);
    [oldLogLik, derX, H_x] = model.likelihood(model, X);
else
   fprintf('error\n')
   return
end

n = length(model.E_mean);

model.D=ones(n,1);
Q_post = model.Q + H_x;
R_post = chol(Q_post);
if model.k==0
        Xmean_ =( X+R_post\(R_post'\(derX - model.Q*X)));
    else
        Xmean_ = R_post\(R_post'\derX );
        
    for iter_=2:model.k
            [newLogLik, derX, H_x]  = model.likelihood(model, Xmean_);
            Q_post = model.Q + H_x;
            R_post = chol(Q_post);
            Xmean_ = R_post\(R_post'\derX);
    end
end


for it = 1:(BurnInIters + Iters) 
    if(mod(it, 1000)==0)
        fprintf('iter = %d (%.2f) (beta = %.2f)\n',it,mean(acceptHistF(1:it-1)), beta);
     end
    %sampling step
    sqrtOneBeta2 = sqrt(1-beta^2);
    
    
    
    Xmean = (1-sqrtOneBeta2) * Xmean_;
    Xnew =   Xmean + sqrtOneBeta2.*X + beta*(R_post\randn( model.n, 1));
     %
    if strcmp(model.adaptive,'eigen')
        [newLogLik, derX_new, H_x_new]  = model.likelihood(model, Xnew);
        
    end
    Q_post_new = model.Q + H_x_new;
    R_post_new = chol(Q_post_new);
    if model.k==0
        Xmean_new_ = ( Xnew+R_post_new\(R_post_new'\(derX_new - model.Q*Xnew)));
    else
        Xmean_new_ = R_post_new\(R_post_new'\derX_new) ;
        for iter_=2:model.k
            [newLogLik, derX_new, H_x_new]  = model.likelihood(model, Xmean_new_);
            Q_post_new = model.Q + H_x_new;
            R_post_new = chol(Q_post_new);
            Xmean_new_ = R_post_new\(R_post_new'\derX_new );
        end
    end
    Xmean_new = (1-sqrtOneBeta2)*Xmean_new_;
    
   
    
    % Metropolis-Hastings to accept-reject the proposal
    xnHxn = (Xnew - sqrtOneBeta2*X)'*H_x *(Xnew - sqrtOneBeta2*X);
    xHx = (X - sqrtOneBeta2*Xnew)'*H_x_new *(X - sqrtOneBeta2*Xnew);
    hxy = newLogLik - oldLogLik + 0.5/beta^2 * (xnHxn - xHx);
    
    munQx   = Xmean_new'*Q_post_new*(X - sqrtOneBeta2 * Xnew - 0.5 * Xmean_new);
    muQxnew = Xmean'*Q_post*(Xnew - sqrtOneBeta2 * X - 0.5 * Xmean);
    hxy = hxy + munQx/beta^2 - muQxnew/beta^2;
    [accept, uprob] = metropolisHastings(hxy, 0, 0, 0); 
   
    acceptHistF(it) = accept;  
     
    if accept == 1
        X = Xnew; 
        oldLogLik = newLogLik; 
        H_x  = H_x_new;
        Q_post = Q_post_new;
        Xmean_ = Xmean_new_;
        R_post = R_post_new;
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
    if (it > BurnInIters)  && (mod(it,StoreEvery) == 0)
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

 
