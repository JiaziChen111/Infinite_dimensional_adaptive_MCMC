
%%
%   sample U, the random field
%
%
%%%    


% Generate the pCN proposal
Zs = randn(n1*n2, 1);
%AMCMC.Z_mean = 0*AMCMC.Z_mean;
c_beta  = sqrt(1-AMCMC.beta^2);
%AMCMC.D = ones(size(AMCMC.D));
if strcmp(method,'CN_cm_cs') %|| ((strcmp(method{2},'apCN') && s < mcmc_adapt))
    Zprop = c_beta * Z +(1-c_beta) * AMCMC.Z_mean + AMCMC.beta*(AMCMC.D.*Zs);
elseif strcmp(method, 'CN') 
    Zprop = c_beta*Z +  AMCMC.beta*Zs;
elseif strcmp(method, 'CNL_cm')
    Zmean = (Z + (AMCMC.D2.*(gradZ(:)- Z))) ;
   Zprop = c_beta * Z +(1-c_beta) * Zmean + AMCMC.beta*(AMCMC.D.*Zs);
elseif strcmp(method, 'MALA')
   Z_m   = Z + 0.5*AMCMC.beta^2*(AMCMC.D2.* (gradZ(:) - Z(:)));
   Zprop =    Z_m + AMCMC.beta*(AMCMC.D.*Zs);
end

% Calculate the acceptance probability aProb
[Uprop] = E(exp(sigma_tau(1)),    ...
            alpha,                ...
            exp(sigma_tau(2)),    ...
            reshape(Zprop, n1, n2), ...
            K1,                   ...
            K2);
[phiUProp, gradUprop] = potential(y, A, Uprop(:), meanY, cellArea);



if strcmp(method,'CN_cm_cs') % || ((strcmp(method{2},'apCN') && s < mcmc_adapt))
   sumQnew = 0.5*sum(AMCMC.Q.*Zprop.^2);
   sumQ    = 0.5*sum(AMCMC.Q.*Z.^2);
    qZ      = - sumQnew + sumQ - (AMCMC.Z_mean./ AMCMC.D2)'*(Zprop-Z) ;
elseif strcmp(method,'CN')
   sumQnew = 0;
   qZ = 0;
elseif strcmp(method,'CNL_cm')
   [gradZprop] = Et(sigma,                     ...
                    alpha,                     ...
                    tau,                       ...
                    reshape(gradUprop, n1, n2),  ...
                    K1,                        ...
                    K2);
   Zmean_new   = Zprop + (AMCMC.D2.*(gradZprop(:)- Zprop));
   qZ =      (1-c_beta) * (Z    -  c_beta * Zprop - 0.5 * (1-c_beta) * Zmean_new)'*(Zmean_new./(AMCMC.beta.^2*AMCMC.D2));
   qZ = qZ - (1-c_beta) * (Zprop - c_beta * Z     - 0.5 * (1-c_beta) * Zmean    )'*(Zmean./(AMCMC.beta.^2*AMCMC.D2));
   sumQnew = 0.5*sum(AMCMC.Q.*Zprop.^2);
   sumQ    = 0.5*sum(AMCMC.Q.*Z.^2);
   qZ = qZ  - sumQnew + sumQ;

elseif strcmp(method,'MALA')
    [gradZprop] = Et(sigma,                      ...
                    alpha,                       ...
                    tau,                         ...
                    reshape(gradUprop, n1, n2),  ...
                    K1,                          ...
                    K2);
   Zprop_m     = Zprop + 0.5*AMCMC.beta^2*(AMCMC.D2.* (gradZprop(:) - Zprop(:)));
   qZ =      0.5 * (Zprop  - Z_m)' * ((Zprop  - Z_m) ./ (AMCMC.beta.^2*AMCMC.D2));
   qZ = qZ - 0.5 * (Z  - Zprop_m)' * ((Z  - Zprop_m) ./ (AMCMC.beta.^2*AMCMC.D2));
   qZ = qZ - 0.5 * (Zprop' * Zprop);
   qZ = qZ + 0.5 * (Z' * Z);
end
if(phiUProp==-Inf)
    aProb = 0;
else

    aProb = min(1,exp(  phiUProp - phiU + qZ ));
end


% Accept proposal with probability aProb
if unifrnd(0,1) < aProb
    Z     = Zprop;
    phiU  = phiUProp;
    U     = Uprop;
    acc(i) = 1;
    if strcmp(method,'CNL_cm')
        gradZ = gradZprop;
        Zmean = Zmean_new;
    elseif strcmp(method,'MALA')
        Z_m = Zprop_m;
    end
    acc_RR = acc_RR +1;

end
%%
%   sample hyperparameter
%
%
%%%
sigma_tau_prop = sigma_tau + beta2*Sigma12_st*randn(2,1);
[Uprop, sqrt_lambda] = E(exp(sigma_tau_prop(1)), ...
            alpha,                            ...
            exp(sigma_tau_prop(2)),           ...
            reshape(Z, n1, n2),                 ...
            K1,                               ...
            K2);
[phi_st_Prop, gradUprop] = potential(y, A, Uprop(:), meanY, cellArea);
sigma_tau_prop_m = sigma_tau_prop - mean_st;
sigma_tau_m = sigma_tau - mean_st;
qZ = -sigma_tau_prop_m'*sigma_tau_prop_m/(2*0.15^2) + sigma_tau_m'*sigma_tau_m/(2*0.15^2);%prior
%qZ = qZ - sum(sigma_tau_prop ) + sum(sigma_tau );
aProb = min(1,exp(  phi_st_Prop - phiU + qZ));
if unifrnd(0,1) < aProb
    phiU             = phi_st_Prop;
    U                = Uprop;
    acc_vec_hyper(i) = 1;
    sigma           = exp(sigma_tau_prop(1));
    tau             = exp(sigma_tau_prop(2));
    sigma_tau       = sigma_tau_prop;
    AMCMC.invLambda = 1./sqrt_lambda.^2;

    if strcmp(method, 'CNL_cm')
        [gradZ]     = Et(sigma,                     ...
                         alpha,                     ...
                         tau,                       ...
                         reshape(gradUprop, n1, n2),  ... 
                         K1,                        ...
                         K2);
        
    elseif strcmp(method, 'MALA')

        [gradZ] = Et(sigma,                     ...
                         alpha,                     ...
                         tau,                       ...
                         reshape(gradUprop, n1, n2),  ... 
                         K1,                        ...
                         K2);
    end
    acc_RR2= acc_RR2 + 1;
end

[beta2,  acc_RR2]  = AMCMC_RR(beta2, acc_RR2, i); 
[beta,  acc_RR]  = AMCMC_RR(beta, acc_RR, i, 50,accepantce_rate, 1);
AMCMC.beta = beta; 

