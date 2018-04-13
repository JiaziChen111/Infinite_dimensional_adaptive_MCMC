function [AMCMC] = AMCMC_D(x, count, AMCMC, I,burnin)
%%
%   adaptive D, Q for pCN
%
%
%
%%

n_adpt = AMCMC.n_adpt;
n  =length(AMCMC.E_var);
if count > burnin && mod(count,1000) == 0
    
    n_adpt = n_adpt + 20;
    AMCMC.n_adpt = min(n_adpt,n);
end

if mod(count, AMCMC.batch) > 0
   return
end
if count > burnin/2
    AMCMC.count       = AMCMC.count + 1;
    w_m               = (1 / AMCMC.count);
    AMCMC.E_mean      = AMCMC.E_mean * (1-w_m)  + w_m * x;
    AMCMC.E2_mean     = AMCMC.E2_mean * (1-w_m) + w_m * x.^2;
    AMCMC.E_var       = AMCMC.E2_mean  - AMCMC.E_mean.^2;
    

    if min(AMCMC.E_var)< 0 || sum(AMCMC.E_var)<10^-14 
        AMCMC.E_var   = ones(size(AMCMC.E_var  ));
    end
end
if count > burnin && n_adpt < n -10 

       index_ = max(n_adpt-9,1):min( (n_adpt+9),n);
       a = AMCMC.a;
       for k=1:4
           D2_ = AMCMC.E_var(index_);
           c = 1./(AMCMC.invLambda(index_) + a) ;
           da = D2_.*c  -AMCMC.invLambda(index_).*c.^2;
           dda = -da.*c + AMCMC.invLambda(index_).*c.^3;
           a = a - 0.01*sum(da)/sum(dda);
       end
       if(isreal(a)==0)
           AMCMC.a = 0;
       else
           AMCMC.a = max(a,0);
       end
       
       AMCMC.D_adapt = sqrt(AMCMC.invLambda./(AMCMC.invLambda + AMCMC.a));


      
end    
    
    
if count > burnin
    AMCMC.Z_mean(I) = [AMCMC.E_mean(I(1:AMCMC.n_adpt))     ;  0*(1-AMCMC.D(I(AMCMC.n_adpt+1:end))).*AMCMC.E_mean(I(AMCMC.n_adpt+1:end))]; 
    AMCMC.D(I(1:AMCMC.n_adpt))       = min(1,sqrt(AMCMC.E_var(I(1:AMCMC.n_adpt))));
    if AMCMC.n_adpt < n-10
        start_point = mean(sqrt(AMCMC.E_var(I((AMCMC.n_adpt-10):AMCMC.n_adpt+1))))/AMCMC.D_adapt(I(AMCMC.n_adpt+1));
        AMCMC.D(I(AMCMC.n_adpt+1:end))   = min(1,start_point * AMCMC.D_adapt(I(AMCMC.n_adpt+1:end)));
    end
    AMCMC.D2  = AMCMC.D.^2;
    AMCMC.Q   = 1 - 1./AMCMC.D2;
end
      

end
    
    