function U = sample_U(Lambda_sqrt)

[n1, n2] = size(Lambda_sqrt);
xi=randn(n1,n2)+1i.*randn(n1,n2); 

U = real(fft2(Lambda_sqrt .* xi))/sqrt(n1*n2);