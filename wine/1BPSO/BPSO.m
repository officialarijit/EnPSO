clc
clear all
close all

%% ////////////////////////////////////////////////////Data set preparation/////////////////////////////////////////////

load Training_wine.mat
x=Training;

[ro,col]=size(x);
for ii=1:(col-1)
    H=x(1:size(x,1),ii)';
    [xf,PS1] = mapminmax(H);%Process matrices by mapping row minimum and maximum values to [-1 1]
    I2(:,ii)=xf;
end
T=x(1:ro,col); %Targeted O/p


Trgt_op=T';
T=T';
[yf,PS5]= mapminmax(T);
T=yf;
T=T';

inpt_inst=size(x,2) -1;
Ono = size(unique(Trgt_op),2);

%% /////////////////////////////////////////////FNN initial parameters//////////////////////////////////////
HiddenNodes=5;       %Number of hidden codes
nw =(inpt_inst*HiddenNodes)+(HiddenNodes*Ono);
nb = (HiddenNodes+Ono);
Dim=nw+nb ; %Dimension of particles in PSO
TrainingNO=size(x,1);       %Number of training samples

%% ////////////////////////////////////////////////////////PSO/////////////////////////////////////////////
%Initial Parameters for PSO
noP=50;           %Number of particles
Max_iteration=500;%Maximum number of iterations

wMax=0.9;         %Max inirtia weight
wMin=0.3;         %Min inirtia weight
c1=1;
c2=1;

vel=zeros(noP,Dim); %Velocity vector
pos=zeros(noP,Dim); %Position vector

%////////Cognitive component/////////
pBestScore=zeros(noP);
pBest=zeros(noP,Dim);
%////////////////////////////////////

%////////Social component///////////
gBestScore=inf;
gBest=zeros(1,Dim);
%///////////////////////////////////

ConvergenceCurve=zeros(1,Max_iteration); %Convergence vector

%Initialization
for i=1:size(pos,1) % For each Particle
    for j=1:size(pos,2) % For each dimension
        pos(i,j)=unifrnd(0,1);
        vel(i,j)=unifrnd(0,1);
    end
end

%initialize gBestScore for min
gBestScore=inf;


for Iteration=1:Max_iteration
    %Calculate MSE
    for i=1:size(pos,1)
        for ww=1:nw
            Weights(ww)=pos(i,ww);
        end
        for bb=nw+1:Dim
            Biases(bb-(nw))=pos(i,bb);
        end
        fitness=0;
        for pp=1:TrainingNO
            tempTraindata=I2(pp,:);
            actualvalue=My_FNN(inpt_inst,HiddenNodes,Ono,Weights,Biases,tempTraindata);
            if(T(pp)==-1)
                fitness=fitness+(1-actualvalue(1))^2;
                fitness=fitness+(0-actualvalue(2))^2;
                fitness=fitness+(0-actualvalue(3))^2;
            end
            if(T(pp)==0)
                fitness=fitness+(0-actualvalue(1))^2;
                fitness=fitness+(1-actualvalue(2))^2;
                fitness=fitness+(0-actualvalue(3))^2;
            end
            if(T(pp)==1)
                fitness=fitness+(0-actualvalue(1))^2;
                fitness=fitness+(0-actualvalue(2))^2;
                fitness=fitness+(1-actualvalue(3))^2;
            end
        end
        fitness=fitness/TrainingNO;
        
        if Iteration==1
            pBestScore(i)=fitness;
        end
        
        if(pBestScore(i)>fitness)
            pBestScore(i)=fitness;
            pBest(i,:)=pos(i,:);
        end
        
        if(gBestScore>fitness)
            gBestScore=fitness;
            gBest=pos(i,:);
        end
        
        if(gBestScore==1)
            break;
        end
    end
    %Update the w of PSO
    w=wMax-Iteration*(wMax-wMin)/Max_iteration;
    
    %Update the velocity and position of particles
    for i=1:size(pos,1)
        for j=1:size(pos,2)
            vel(i,j)=w*vel(i,j)+c1*rand()*(pBest(i,j)-pos(i,j))+c2*rand()*(gBest(j)-pos(i,j));
            pos(i,j)=pos(i,j)+vel(i,j);
        end
    end
    ConvergenceCurve(1,Iteration)=gBestScore;
    
    disp(['PSO is training FNN (Iteration = ', num2str(Iteration),' ,MSE = ', num2str(gBestScore),')'])
end

%% ---------------------- Testing Starts Here----------------------------------
%% ///////////////////////Calculate the classification//////////////////////
load Testing_wine.mat
load Actual_op.mat
x=Testing;
%% ///////////////////////Calculate the classification/////////////////////

disp('------------------TESTING STARTS-----------------------------------')

%Process matrices by mapping row minimum and maximum values to [-1 1]
r=size(x,1);
c=size(x,2);
for ii=1:(c-1)
    H=x(1:size(x,1),ii)';
    [xf,PS1] = mapminmax(H);
    I3(:,ii)=xf;
end
T=x(1:size(x,1),c); %Targeted O/p

Trgt_op=T';
T=T';
[yf,PS5]= mapminmax(T);
T=yf;
T=T';

inpt_inst=size(x,2) -1;
Ono = size(unique(Trgt_op),2);

%% /////////////////////////////////////////////FNN initial parameters//////////////////////////////////////
HiddenNodes=5;       %Number of hidden codes
nw =(inpt_inst*HiddenNodes)+(HiddenNodes*Ono);
nb = (HiddenNodes+Ono);
Dim=nw+nb ; %Dimension of particles in PSO
TestingingNO=size(x,1);       %Number of training samples

Rrate=0;
Weights=gBest(1:nw);
Biases=gBest(nw+1:Dim);
for pp=1:TestingingNO
    tempTraindata=I3(pp,:);
    actualvalue=My_FNN(inpt_inst,HiddenNodes,Ono,Weights,Biases,tempTraindata);
    FNNProduced_op(pp,:)=actualvalue;
    Prod_op(pp,:)=round([actualvalue(1)  actualvalue(2) actualvalue(3)]);
    %     if(T(pp)==-1)
    %         if (round(actualvalue(1))==1 && round(actualvalue(2))==0 && round(actualvalue(3))==0)
    %             Rrate=Rrate+1;
    %         end
    %     end
    %     if(T(pp)==0)
    %         if (round(actualvalue(1))==0 && round(actualvalue(2))==1 && round(actualvalue(3))==0)
    %             Rrate=Rrate+1;
    %         end
    %     end
    %     if(T(pp)==1)
    %         if (round(actualvalue(1))==0 && round(actualvalue(2))==0 && round(actualvalue(3))==1)
    %             Rrate=Rrate+1;
    %         end
    %     end
end

% ClassificationRate=(Rrate/TestingingNO)*100;
% disp(['Classification rate = ', num2str(ClassificationRate) '%'] );

%=========confusion Matrix Calculation=========%
ro=size(x,1);
col=size(x,2);
for ii=1:ro
    temp = Prod_op(ii,:);
    if temp == Actual_op(ii,:)
        shi(ii)=1;
    else
        shi(ii)=0;
    end
    if(temp(1,1) == 1 && temp(1,2) == 0 && temp(1,3) == 0)
        NN_op(ii)=1;
        
    elseif (temp(1,1) == 0 && temp(1,2) == 1 && temp(1,3) == 0)
        NN_op(ii)=2;
        
    elseif(temp(1,1) == 0 && temp(1,2) == 0 && temp(1,3) == 1 )
        NN_op(ii)=3;
        
    else
        NN_op(ii)=0;
        
    end
end

[fmissclass,C,ind,per] = confusion(Actual_op',FNNProduced_op');
Cval=Evaluate(Trgt_op,NN_op);

PCCE=(sum(shi)/TrainingNO)*100; %percentage of correctly classified examples

MSE=mse(Actual_op , FNNProduced_op);

RMSE = sqrt(MSE);

NRMSE = (RMSE/(sum(Trgt_op)/TrainingNO))*100;

disp(['accuracy:' num2str(Cval(1,1)*100) '%']) ;
disp(['sensitivity:' num2str(Cval(1,2))  ]);
disp(['specificity:' num2str(Cval(1,3)) ]);
disp(['precision' num2str(Cval(1,4))  ]);
disp(['recall:'  num2str(Cval(1,5))  ]);
disp(['f_measure:' num2str(Cval(1,6)) ]) ;
disp(['gmean:' num2str(Cval(1,7))  ]);
disp(['Means square error:' num2str(MSE)  ]);
disp(['Root means square error:' num2str(RMSE)  ]);
disp(['percentage of correctly classified example(PCCE):' num2str(PCCE) '%' ]);
disp(['Normalized root means square error:' num2str(NRMSE)  ]);
Cval=[Cval MSE RMSE PCCE NRMSE]
%===================================================
ClassificationRate=Cval(1,1)*100;
if ClassificationRate > 50
    
    t1 = datetime('now');
    t1=char(t1);
    strn = regexprep(t1,'[- :]','_');
    
    %% Draw the convergence curve
    figure(1)
    hold on;
    semilogy(ConvergenceCurve);
    title(['Classification rate : ', num2str(ClassificationRate), '%']);
    xlabel('Iteration');
    ylabel('MSE');
    box on
    grid on
    axis tight
    hold off;
    
    %% Regression Plot
    figure(2)
    plotregression(Actual_op,FNNProduced_op);
    [r,m,b] = regression(Actual_op,FNNProduced_op);
    disp(['The regression value is:' num2str(mean(r))]);
    
    %% Plot receiver operating characteristic
    figure(3)
    plotroc(Actual_op',FNNProduced_op');
    
    %% Confusion Matrix Print
    disp('Confusion Matrix:')
    disp(C)
    disp(['Confusion Val(Fraction of misclass):' num2str(fmissclass) ] );
    
    %% plotting confusion matrix
    figure(4)
    plotconfusion(Actual_op',FNNProduced_op')
    
    %% Saving Files
    filename=[strn 'FNNOPSO_wine POP ' num2str(noP) ' C rate' num2str(ClassificationRate) '.mat'];
    filename1=[strn 'FNNOPSO_wine POP ' num2str(noP) ' C rate' num2str(ClassificationRate) ' CPlot.png'];
    filename2=[strn 'FNNOPSO_wine POP ' num2str(noP) ' C rate' num2str(ClassificationRate) ' Rplot.png'];
    filename3=[strn 'FNNOPSO_wine POP ' num2str(noP) ' C rate' num2str(ClassificationRate) ' ROCplot.png'];
    filename4=[strn 'FNNOPSO_wine POP ' num2str(noP) ' C rate' num2str(ClassificationRate) ' ConfusionMat plot.png'];
    save(filename);%Saving workspace
    print(figure(1),filename1,'-dpng')
    print(figure(2),filename2,'-dpng')
    print(figure(3),filename3,'-dpng')
    print(figure(4),filename4,'-dpng')
end