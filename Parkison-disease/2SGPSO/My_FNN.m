function o=My_FNN(Ino,Hno,Ono,W,B,tempTraindata)
x=tempTraindata;
col=size(x,2);

h=zeros(1,Hno);
o=zeros(1,Ono);
temp=0;

% for i=1:Hno
%     h(i)=My_sigmoid(x1*W(0*Hno+i)+x2*W(Hno+i)+x3*W(2*Hno+i)+x4*W(3*Hno+i)+x5*W(4*Hno+i)...
%         +x6*W(5*Hno+i)+x7*W(6*Hno+i)+x8*W(7*Hno+i)+x9*W(8*Hno+i)+x10*W(9*Hno+i)+...
%         x10*W(9*Hno+i)+x11*W(10*Hno+i)+x12*W(11*Hno+i)+x13*W(12*Hno+i)+B(i));
% end

for i=1:Hno
    for jj=1:col
        temp=temp+x(jj)*W((jj-1)*Hno+i)  ;
    end
    val=My_sigmoid(temp+B(i));
    h(i)=val;
end

k=3;
for i=1:Ono
    k=k+1;
    for j=1:Hno
        o(i)=o(i)+(h(j)*W(k*Hno+j));
    end
end
for i=1:Ono 
    o(i)=My_sigmoid(o(i)+B(Hno+i));
end

