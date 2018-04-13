% not updated

if(0)
%parameter proposal
sigma_tau_prop = sigma_tau + AMCMC.beta*Sigma12_st*randn(2,1);
    
Zs = randn(length(Z), 1);
c_beta  = sqrt(1-AMCMC.beta^2);
if strcmp(type,'cspCN') 
Zprop = c_beta * Z +(1-c_beta) * AMCMC.Z_mean + AMCMC.beta*(AMCMC.D.*Zs);
elseif strcmp(type,'pCN')
Zprop = c_beta * Z +  AMCMC.beta * Zs;
elseif strcmp(type,'apCN')
           [Uprop_param] = E(exp(sigma_tau_prop(1)), ...
                             alpha,                  ...
                             exp(sigma_tau_prop(2)), ...
                             reshape(Z, n1, n2),     ...
                             K1, K2);
           [phiUprop_1, gradUprop_param] = potential(y, A, Uprop_param(:), meanY, cellArea);
           [gradZprop] = Et(exp(sigma_tau_prop(1)), ...
                            alpha,                  ...
                            exp(sigma_tau_prop(2)), ...
                            reshape(gradUprop_param, n1, n2), ...
                            K1, K2);
           Zmean_prop = (Z + (AMCMC.D2.*(gradZprop(:)- Z)));
           Zprop = c_beta * Z +(1-c_beta) * Zmean_prop + AMCMC.beta*(AMCMC.D.*Zs); 
            % Calculate the acceptance probability aProb
            [Uprop] = E(exp(sigma_tau(1)),...
                        alpha, ...
                        exp(sigma_tau(2)), ...
                        reshape(Zprop, n1, n2), ...
                        K1, K2);
            [phiUProp_2, gradUprop] = potential(y, A, Uprop(:), meanY, cellArea);
            [gradZprop] = Et(exp(sigma_tau(1)), ...
                             alpha,             ...
                             exp(sigma_tau(2)), ...
                             reshape(gradUprop, n1, n2), ...
                             K1, K2);
            Zmean_new   = Zprop + (AMCMC.D2.*(gradZprop(:)- Zprop));
end
           
 [Uprop,sqrt_lambda]  = E(exp(sigma_tau_prop(1)), ...
                    alpha, ...
                    exp(sigma_tau_prop(2)),...
                    reshape(Zprop, n1, n2), K1, K2);
    phiUProp = potential(y, A, Uprop(:), meanY, cellArea);

if strcmp(type,'cspCN')
       sumQnew = 0.5*sum(AMCMC.Q.*Zprop.^2);
       sumQ    = 0.5*sum(AMCMC.Q.*Z.^2);
       qZ      = - sumQnew + sumQ - ( AMCMC.D2.*AMCMC.Z_mean)'*(Zprop-Z) ;
elseif strcmp(type,'pCN') 
       qZ = 0;
elseif strcmp(type,'apCN')
           qZ =      (1-c_beta) * (Z    -  c_beta * Zprop - 0.5 * (1-c_beta) * Zmean_new )'*(Zmean_new./(AMCMC.beta.^2*AMCMC.D2));
           qZ = qZ - (1-c_beta) * (Zprop - c_beta * Z     - 0.5 * (1-c_beta) * Zmean_prop)'*(Zmean_prop./(AMCMC.beta.^2*AMCMC.D2));
           sumQnew = 0.5*sum(AMCMC.Q.*Zprop.^2);
           sumQ    = 0.5*sum(AMCMC.Q.*Z.^2);
           qZ = qZ  - sumQnew + sumQ;
end

sigma_tau_prop_m = sigma_tau_prop - [0;log(10)];
sigma_tau_m = sigma_tau - [0;log(10)];
qZ = qZ -sigma_tau_prop_m'*sigma_tau_prop_m/(2*0.15^2) + sigma_tau_m'*sigma_tau_m/(2*0.15^2);%prior
            
MH_rat = phiUProp - phiU + qZ;
if rand < exp(MH_rat)
   Z = Zprop;
   phiU = phiUProp;
   U = Uprop;
   acc(i ) = 1;
   AMCMC.invLambda = 1./sqrt_lambda.^2;
   acc_RR = acc_RR + 1;
   sigma_tau = sigma_tau_prop;
end
    
    
[beta,  acc_RR]  = AMCMC_RR(beta, acc_RR, i, 50,accepantce_rate);
AMCMC.beta = beta; 
end