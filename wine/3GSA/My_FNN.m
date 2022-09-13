function o=My_FNN(Ino,Hno,Ono,W,B,tempTraindata)
x=tempTraindata;
col=size(x,2);

h=zeros(1,Hno);
o=zeros(1,Ono);
temp=0;

for i=1:Hno
    for jj=1:col
        temp=temp+x(jj)*W((jj-1)*Hno+i)  ;
    end
    val=My_sigmoid(temp+B(i));
    h(i)=val;
end

k=Hno+i;
for i=1:Ono
    k=k+1;
    for j=1:Hno
        o(i)=o(i)+(h(j)*W(k+j));
    end
end
for i=1:Ono 
    o(i)=My_sigmoid(o(i)+B(Hno+i));
end

