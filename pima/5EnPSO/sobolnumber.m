function x1 = sobolnumber(LB1,UB1,nVar)
    N=1;           % number of sample points    
    p = sobolset(nVar,'Skip',1e3,'Leap',1e2);
    p = scramble(p,'MatousekAffineOwen');
    A=net(p,N);
    %%% The generated numbers are transferred to their correct range of parameters and rounded to two decimal places %%%
    x1=((LB1+(UB1-LB1).*A(:,1))*100)/100;
end

