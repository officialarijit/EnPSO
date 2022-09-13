%% ----------------------------------------------------------------------------
% PSOGSA source codes version 1.0.
% Author: Seyedali Mirjalili (ali.mirjalili@gmail.com)

% Main paper:
% S. Mirjalili, S. Z. Mohd Hashim, and H. Moradian Sardroudi, "Training
%feedforward neural networks using hybrid particle swarm optimization and
%gravitational search algorithm," Applied Mathematics and Computation,
%vol. 218, pp. 11125-11137, 2012.

%The paper of the PSOGSA algorithm utilized as the trainer:
%S. Mirjalili and S. Z. Mohd Hashim, "A New Hybrid PSOGSA Algorithm for
%Function Optimization," in International Conference on Computer and Information
%Application?ICCIA 2010), 2010, pp. 374-377.
%% -----------------------------------------------------------------------------

clc
clear all
close all

%% ////////////////////////////////////////////////////Data set preparation/////////////////////////////////////////////
load Training_heart.mat
x=Training;

%Process matrices by mapping row minimum and maximum values to [-1 1]
[r,c]=size(x);
for ii=1:(c-1)
    H=x(1:size(x,1),ii)';
    [xf,PS1] = mapminmax(H);
    I2(:,ii)=xf;
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
TrainingNO=size(x,1);       %Number of training samples

%% ////////////////////////////////////////////////////////GSA/////////////////////////////////////////////

%Configurations and initializations

noP = 50;             %Number of masses
Max_iteration  = 500;  %Maximum number of iteration
w=2;              %Inirtia weight
wMax=0.9;         %Max inirtia weight
wMin=0.5;         %Min inirtia weight

CurrentFitness =zeros(noP,1);

G0=1; %Gravitational constant
CurrentPosition = rand(noP,Dim); %Postition vector
Velocity = .3*randn(noP,Dim) ; %Velocity vector
acceleration=zeros(noP,Dim); %Acceleration vector
mass(noP)=0; %Mass vector
force=zeros(noP,Dim);%Force vector

%Vectores for saving the location and MSE of the best mass
gBestScore=inf;
gBest=zeros(1,Dim);


ConvergenceCurve=zeros(1,Max_iteration); %Convergence vector

%Main loop
Iteration = 0 ;
while  ( Iteration < Max_iteration )
    Iteration = Iteration + 1;
    G=G0*exp(-20*Iteration/Max_iteration); %Equation (3.3)
    force=zeros(noP,Dim);
    mass(noP)=0;
    acceleration=zeros(noP,Dim);
    
    %Calculate MSEs
    
    for i = 1:noP
        for ww=1:(nw)
            Weights(ww)=CurrentPosition(i,ww);
        end
        for bb=nw+1:Dim
            Biases(bb-(nw))=CurrentPosition(i,bb);
        end
        fitness=0;
        for pp=1:TrainingNO
            tempTraindata=I2(pp,:);
            actualvalue=My_FNN(inpt_inst,HiddenNodes,Ono,Weights,Biases,tempTraindata);
            
            if(T(pp)==-1)
                fitness=fitness+(1-actualvalue(1))^2;
                fitness=fitness+(0-actualvalue(2))^2;
                
            end
            
            if(T(pp)==1)
                fitness=fitness+(0-actualvalue(1))^2;
                fitness=fitness+(1-actualvalue(2))^2;
            end
        end
        fitness=fitness/TrainingNO; %Equation (5.4)
        CurrentFitness(i) = fitness;
        
        if(gBestScore>fitness)
            gBestScore=fitness;
            gBest=CurrentPosition(i,:);
        end
    end
    
    best=min(CurrentFitness);%Equation (3.10)
    worst=max(CurrentFitness);%Equation (3.11)
    
    for i=1:noP
        mass(i)=(CurrentFitness(i)-0.99*worst)/(best-worst);%Equation (3.9)
    end
    
    for i=1:noP
        mass(i)=mass(i)*5/sum(mass);%Equation (3.14)
        
    end
    
    %Calculate froces
    
    for i=1:noP
        for j=1:Dim
            for k=1:noP
                if(CurrentPosition(k,j)~=CurrentPosition(i,j))
                    %Equation (3.5)
                    force(i,j)=force(i,j)+ rand()*G*mass(k)*mass(i)*(CurrentPosition(k,j)-CurrentPosition(i,j))/abs(CurrentPosition(k,j)-CurrentPosition(i,j));
                    
                end
            end
        end
    end
    
    %Calculate a
    
    for i=1:noP
        for j=1:Dim
            if(mass(i)~=0)
                acceleration(i,j)=force(i,j)/mass(i);%Equation (3.6)
            end
        end
    end
    
    %Update inertia weight
    
    w=wMin-Iteration*(wMax-wMin)/Max_iteration;
    
    %Calculate V
    for i=1:noP
        for j=1:Dim
            %Equation (4.1)
            Velocity(i,j)=w*Velocity(i,j)+rand()*acceleration(i,j) + rand()*(gBest(j)-CurrentPosition(i,j));
        end
    end
    
    %Calculate X
    
    CurrentPosition = CurrentPosition + Velocity ; %Equation (4.2)
    
    ConvergenceCurve(1,Iteration)=gBestScore;
    disp(['PSOGSA is training FNN (Iteration = ', num2str(Iteration),' ,MSE = ', num2str(gBestScore),')'])
    
end
%% ///////////////Classification Testing data ready ///////////////////%
load Testing_heart.mat
load Actual_op.mat
x=Testing;
%% ///////////////////////Calculate the classification/////////////////////

disp('------------------TESTING STARTS-----------------------------------')

%Process matrices by mapping row minimum and maximum values to [-1 1]
[r,c]=size(x);
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
    Prod_op(pp,:)=round([actualvalue(1)  actualvalue(2)]);
    if(T(pp)==-1)
        if (round(actualvalue(1))==1 && round(actualvalue(2))==0 )
            Rrate=Rrate+1;
        end
    end
    
    if(T(pp)==1)
        if (round(actualvalue(1))==0 && round(actualvalue(2))==1)
            Rrate=Rrate+1;
        end
    end
end
% ClassificationRate=(Rrate/TrainingNO)*100;
% classerr=(1-(Rrate/TrainingNO))*100;
% disp(['Classification rate = ', num2str(ClassificationRate) '%']);
% disp(['Classification error rate = ', num2str(classerr) '%'] );
% disp('-------------------------------------------------------------------------------')
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
    if temp == Actual_op(ii,:)
        shi(ii)=1;
    else
        shi(ii)=0;
    end
    if(temp(1,1) == 1 && temp(1,2) == 0  )
        NN_op(ii)=1;
    elseif (temp(1,1) == 0 && temp(1,2) == 1 )
        NN_op(ii)=2;
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
%==============================================%
ClassificationRate=(Cval(1,1)*100);

if ClassificationRate >80
    
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
    filename=[strn 'FNNPSOGSA_heart POP ' num2str(noP) ' C rate' num2str(ClassificationRate) '.mat'];
    filename1=[strn 'FNNPSOGSA_heart POP ' num2str(noP) ' C rate' num2str(ClassificationRate) ' CPlot.png'];
    filename2=[strn 'FNNPSOGSA_heart POP ' num2str(noP) ' C rate' num2str(ClassificationRate) ' Rplot.png'];
    filename3=[strn 'FNNPSOGSA_heart POP ' num2str(noP) ' C rate' num2str(ClassificationRate) ' ROCplot.png'];
    filename4=[strn 'FNNPSOGSA_heart POP ' num2str(noP) ' C rate' num2str(ClassificationRate) ' ConfusionMat plot.png'];
    save(filename);%Saving workspace
    print(figure(1),filename1,'-dpng')
    print(figure(2),filename2,'-dpng')
    print(figure(3),filename3,'-dpng')
    print(figure(4),filename4,'-dpng')
end