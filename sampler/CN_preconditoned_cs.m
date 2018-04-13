function [model, samples, accRates] = CN_preconditoned_cs(model, mcmcoptions)
%%
%
% Crank Nichlson for latent Gaussian models,
% adaptive estimating, scale and gradient
% using the preconditioned approach  
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
epsilon = 0.5;
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
E_mean = model.E_mean;
%n_adpt = 10;

n_adpt=  15;
a = 0.;
model.D_apt=ones(n,1);
 model.A=  model.A*model.U * diag(sqrt(model.Lambda));
for it = 1:(BurnInIters + Iters) 
%     
    if(mod(it, 10000)==0)
        beta = sqrt(1- (2 / (2 + model.delta))^2) ; 
        fprintf('iter = %d (%.0f) (beta = %.2f, delta = %.2f)\n',it,mean(acceptHistF((it-49):it))*100,beta, model.delta);
        
    end
     %model.D = ones(n,1);
    %sampling step
    C_ = 1 ./(1 + 0.5 * model.delta * model.D.^2);
    C0 = sqrt(2 * model.delta) * C_  .* model.D;
    C1 = C_.*(1 - 0.5 * model.delta * model.D.^2);
    %EC = (1-C1) .* E_mean;
    Enew =   (1-C1) .*E_mean + C1.*E + (C0.*randn( model.n, 1));
    %
    if strcmp(model.adaptive,'eigen')
        Xnew=  Enew;
    elseif strcmp(model.adaptive,'chol')
        Xnew  = model.C_chol* Enew;
    elseif strcmp(model.adaptive,'invchol')
        Xnew  = model.Cinv_chol'\Enew;
    end
    
    newLogLik = model.likelihood(model, Xnew);
     
    % Metropolis-Hastings to accept-reject the proposal
     hyx = oldLogLik +  (E_mean)'*(Enew-E);
     hxy = newLogLik;
     
    [accept, uprob] = metropolisHastings(hxy, hyx, 0, 0); 
   
    acceptHistF(it) = accept;  
     
    if accept == 1
        X = Xnew;
        E = Enew;   
        oldLogLik = newLogLik; 
    end
    
     if n_adpt < n && mod(it,50) == 0 && it >= BurnInIters/2 
        
           if( n_adpt+9 < n)
               index_ = (n_adpt-9):(n_adpt+9);
               for k=1:4
                   D2_ = model.D(index_).^2;
                   c = 1./(model.invLambda(index_) + a) ;
                   da = D2_.*c  -model.invLambda(index_).*c.^2;
                   dda = -da.*c + model.invLambda(index_).*c.^3;
                   a = a - 0.01*sum(da)/sum(dda);
               end
           end
           model.D_apt = sqrt(model.invLambda./(model.invLambda + a));
            model.D_apt(1:n_adpt) = model.D(1:n_adpt);
          if(it <= 3*BurnInIters/4)
              D = model.D_apt;
          end
     end
    
    if(it == BurnInIters)
        opt = 0.3;
    end
    if mod(it,1000) == 0 && (it >= BurnInIters) 
       n_adpt =  n_adpt + 10;
       n_adpt = min(n_adpt, n);
       %plot(D1)
       %drawnow;
    end
    % Adapt proposal during burnin
    if mod(it,10) == 0 
        
        
       
        if (it >= BurnInIters/4)
            model.addapt_count_mean = model.addapt_count_mean  + 1;
            model.addapt_count = model.addapt_count + 1;
            w_m = (1 / model.addapt_count_mean);
            w   = (1 / model.addapt_count);
            model.E_mean  = model.E_mean * (1-w_m) + w_m * E;
            model.D  = min(1,sqrt(model.D.^2 * (1-w)  + w * (E - model.E_mean).^2));
            
            if(it >=   3*BurnInIters/4)
                E_mean = [model.E_mean(1:n_adpt); zeros(n - n_adpt,1)];
            end
                
            
        end
        if (it >= 50) 
            accRateF = mean(acceptHistF((it-49):it))*100;
                if (accRateF > (100*(opt+range))) || (accRateF < (100*(opt-range)))
                 
                    count_m = count_m + 1;
                    epsilon_n = sqrt(epsilon^2 /count_m); 
                     model.delta = model.delta *exp((epsilon_n*((accRateF/100 - opt))));
                   
                end
        end
    end
    
    % keep samples after burn in
    if (it > BurnInIters)  && (mod(it,StoreEvery) == 0)
    %
        cnt = cnt + 1;
        samples.X(cnt,:) = X;  
        samples.E(cnt,:) = E;
        samples.beta_vec(cnt) = beta;
    end
    %    
    samples.LogL(it) = oldLogLik;
    samples.ff(it) = 0.5*(X'*X);    
end
%
%
model.X = X;
accRates.X = mean(acceptHistF(BurnInIters+1:end))*100; 

 
