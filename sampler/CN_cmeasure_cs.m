function [model, samples, accRates] = CN_cmeasure_cs(model, mcmcoptions)
%%
%
%   Crank Nicholson method, where one changes the base measure adaptivly
%   and replaced gradient with posterior mean (Expected gradient)
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
% mod: Jonas Wallin (2018)
%%

BurnInIters = mcmcoptions.Burnin; 
Iters = mcmcoptions.T; 
StoreEvery = mcmcoptions.StoreEvery;
n = length(model.X);
num_stored = floor(Iters/StoreEvery);
X = model.X;
samples.X = zeros(num_stored, n);
samples.E = zeros(num_stored, n);
samples.LogL = zeros(1, num_stored);
samples.num_calls = 0;
samples.beta_vec = zeros(num_stored, 1);


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
epsilon = 0.1;
count_m=1;

acceptHistF = zeros(1, BurnInIters + Iters);

 if strcmp(model.adaptive,'eigen')
        E  = model.U' * ( sqrt(model.invLambda).* X);
 elseif strcmp(model.adaptive,'chol')
        E  = model.C_chol\ X;
 elseif strcmp(model.adaptive,'invchol')
        E  = model.Cinv_chol'*X;
 else
    fprintf('error\n'); 
    return;
 end

n = length(model.E_mean);
model.DX  = zeros(size(model.D));
model.X_mean  = zeros(size(model.E_mean));
D2  = 1./model.D.^2;
Q   = 1 - D2;
D = model.D;
E_mean = model.E_mean;
model.V_mean = zeros(size(model.E_mean));
ED2 = E_mean.*D2;

n_adpt=  50;
sumQ = 0.5*sum(Q.*E.^2);
a = 0.;
model.D_apt=ones(n,1);
Us = model.U * diag(sqrt(model.Lambda));
model.A= model.A*Us;
Us = 1;
for it = 1:(BurnInIters + Iters) 
%     
    if(mod(it, 10000)==0)
        fprintf('iter = %d (%.2f) (beta = %.2f)\n',it,mean(acceptHistF(1:it-1)), beta);
    end
     %model.D = ones(n,1);
    %sampling step
    sqrtOneBeta = sqrt(1-beta.^2);
    Enew =   (1-sqrtOneBeta) .* E_mean + sqrtOneBeta.*E + beta.*(D.*randn( model.n, 1));
    %
    if strcmp(model.adaptive,'eigen')
        Xnew=   Us * Enew;
    elseif strcmp(model.adaptive,'chol')
        Xnew  = model.C_chol* Enew;
    elseif strcmp(model.adaptive,'invchol')
        Xnew  = model.Cinv_chol'\Enew;
    end
    
    newLogLik = model.likelihood(model, Xnew);
     
    % Metropolis-Hastings to accept-reject the proposal
     hyx = oldLogLik + ED2'*(Enew-E);
     sumQnew = 0.5*sum(Q.*Enew.^2);
     hxy = newLogLik  - sumQnew + sumQ;
     
    [accept, uprob] = metropolisHastings(hxy, hyx, 0, 0); 
   
    acceptHistF(it) = accept;  
     
    if accept == 1
        X = Xnew;
        E = Enew;   
        oldLogLik = newLogLik; 
        sumQ = sumQnew;
    end
    
%      if n_adpt < n && mod(it,50) == 0 && it >= BurnInIters/2 
%         
%            if( n_adpt+9 < n)
%                index_ = (n_adpt-5):(n_adpt);
%                for k=1:4
%                    D2_ = model.D(index_).^2;
%                    c = 1./(model.invLambda(index_) + a) ;
%                    da = D2_.*c  -model.invLambda(index_).*c.^2;
%                    dda = -da.*c + model.invLambda(index_).*c.^3;
%                    a = a - 0.01*sum(da)/sum(dda);
%                end
%            end
%            model.D_apt = sqrt(model.invLambda./(model.invLambda + a));
%             model.D_apt(1:n_adpt) = model.D(1:n_adpt);
%           if(it <= 3*BurnInIters/4)
%               D = model.D_apt;
%               Q   = 1 - 1./D2;
%               D2 = model.D.^2;
%               sumQ = 0.5*sum(Q.*E.^2);
%           end
%      end
%     
    if mod(it,1000) == 0 && (it >= BurnInIters) 
       n_adpt =  n_adpt + 15;
       n_adpt = min(n_adpt, n);
       %plot(D1)
       %drawnow;
    end
    % Adapt proposal during burnin
    if mod(it,5) == 0 
        
        
       
        if (it >= BurnInIters/4)
            model.addapt_count_mean = model.addapt_count_mean  + 1;
            model.addapt_count = model.addapt_count + 1;
            w_m = (1 / model.addapt_count_mean);
            w   = (1 / model.addapt_count);
            model.E_mean  = model.E_mean * (1-w_m) + w_m * E;
            model.V_mean = model.V_mean * (1-w)  + w * (E - model.E_mean).^2;
            model.D  = sqrt( model.V_mean);%min(1,sqrt( model.V_mean));
            
            if(it >=   3*BurnInIters/4)
                E_mean = [model.E_mean(1:n_adpt); zeros(n - n_adpt,1)];
                D = [model.D(1:n_adpt); ones(n - n_adpt,1)];
                %D = [model.D(1:n_adpt); model.D_apt((n_adpt+1):end)];
                D2  = 1./D.^2;
                Q   = 1 - D2;
                ED2 = D2.*E_mean;
                sumQ = 0.5*sum(Q.*E.^2);
            end
                
            
        end
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
        %samples.E(cnt,:) = E;
        samples.beta_vec(cnt) = beta;
    end
    %    
    samples.LogL(it) = oldLogLik;
    %samples.ff(it) = 0.5*(X'*X);    
end
%
%
model.X = X;
accRates.X = mean(acceptHistF(BurnInIters+1:end))*100; 

 
