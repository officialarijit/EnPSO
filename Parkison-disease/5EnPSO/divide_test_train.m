clear all
clc
% load wine.mat
data=csvread('bupa_liver.csv');
A=data;
[m,n] = size(A) ;
P = 0.80 ;
idx = randperm(m)  ;
Training = A(idx(1:round(P*m)),:) ; 
Testing = A(idx(round(P*m)+1:end),:) ;
% [r,c]=size(Testing);
% for i=1:r
%     temp=Testing(i,c);
%     if(temp==1)
%         Actual_op(i,:)=[1,0,0];
%     elseif (temp==2)
%         Actual_op(i,:)=[0,1,0];
%     elseif(temp==3)
%         Actual_op(i,:)=[0,0,1];
%     end
% end