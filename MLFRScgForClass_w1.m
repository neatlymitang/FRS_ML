
% [feature_slct,feature_dependfirst,feature_dependlast,TTpes]=fuzzy_pes(data,target,over);

function [bestacc,besto,bestg,feature,cg,X,Y] = MLFRScgForClass_w1(train_data,train_target,test_data,test_target,omin,omax,gmin,gmax,ostep,gstep)
%SVMcg cross validation by faruto

%%
% by faruto
%Email:patrick.lee@foxmail.com QQ:516667408 http://blog.sina.com.cn/faruto BNU
%last modified 2010.01.17

%% 若转载请注明：
% faruto and liyang , LIBSVM-farutoUltimateVersion 
% a toolbox with implements for support vector machines based on libsvm, 2009. 
% 
% Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for
% support vector machines, 2001. Software available at
% http://www.csie.ntu.edu.tw/~cjlin/libsvm

%% about the parameters of SVMcg 

if nargin < 8
    ostep = 0.0002;
    gstep = 0.005;
end

if nargin < 7
    gmax = 1;
    gmin = 0.001;
end
if nargin < 5
    omax = 0.001;
    omin = 0.0005;
end
%% X:c Y:g cg:CVaccuracy
[X,Y] = meshgrid(omax:-ostep:omin,gmin:gstep:gmax);
[m,n] = size(X);
cg = zeros(m,n);


%% record acc with different c & g,and find the bestacc with the smallest c
besto = 0;
bestg = 0.1;
bestacc = 0;
% basenum = 2;
for i = 1:m
    for j = 1:n
%         cmd = [ basenum^X(i,j) , basenum^Y(i,j) ];
        cmd = [ X(i,j) , Y(i,j) ];
%         feature_slct=fuzzy_pes(train_data,train_target,cmd);
         feature_slct=fuzzy_pes_w(train_data,train_target,cmd);
        feature{i,j}=feature_slct;
        [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data(:,feature_slct),train_target,10,1);
        cg(i,j)=MLKNN_test(train_data(:,feature_slct),train_target,test_data(:,feature_slct),test_target,10,Prior,PriorN,Cond,CondN);
        
%         if cg(i,j) < 1
%             continue;
%         end
%         
%         if cg(i,j) > bestacc
%             bestacc = cg(i,j);
%             besto = X(i,j);
%             bestg = Y(i,j);
%         end        
%         
%         if abs( cg(i,j)-bestacc )<=eps && besto > X(i,j) 
%             bestacc = cg(i,j);
%             besto = X(i,j);
%             bestg = Y(i,j);
%         end        
        
    end
end

end
    
    

function feature_slct=fuzzy_pes_w(data,target,cmd)
%[feature_slct,feature_dependfirst,feature_dependlast,TTopt]=fuzzy_opt(data,target,over)

%%%乐观算法 feature_dependfirst
%%%data归一化后的特征空间
%%%target标记为1和-1的标记空间
%st=cputime;  
over=cmd(1);
k=cmd(2);
target=target';
feature_slct=[];%选出特征
[n,m]=size(data);
[n,label]=size(target);
feature_lft=1:m;
num_cur=0;
stand=0;
TTpes=cell(1,1);
while num_cur<m
    max_depend=0;
     feature_depend=zeros(1,length(feature_lft));
    for j=1:length(feature_lft)
        store=ones(n,label);
        feature=[feature_slct feature_lft(j)];
        data0=data(:,[feature]);   
        mm=size(data0,2);
      
        for L=1:label
            X=find(target(:,L)==1);%X表示在该L标记下为1的样本序号
            Y=find(target(:,L)~=1);%Y表示在该L标记下不为1的样本序号
            x=data0(X,:);%x表示在L标记下为1的新数据集
            y=data0(Y,:);%y表示在L标记下部位1的新数据集
             dis_M=exp((2 * x * y' - repmat(sqrt(sum(x .^ 2, 2) .^ 2), 1, size(y, 1)) - repmat(sqrt(sum(y .^ 2, 2)' .^ 2), size(x, 1), 1)) / k);
            dis_H=exp((2 * x * x' - repmat(sqrt(sum(x .^ 2, 2) .^ 2), 1, size(x, 1)) - repmat(sqrt(sum(x .^ 2, 2)' .^ 2), size(x, 1), 1)) / k);
           % dis=kernel(x,y,'linear');
%             or=ones(length(X),length(Y));
%             kernel_matrix=or-dis;
            %min_hete=zeros(length(X),label);
            for i=1:length(X)
                temp=max(dis_M(i,:))-min(dis_H(i,:));                
                if temp<store(X(i),L)
                    store(X(i),L)=temp;
                end
            end
        seq=zeros(n,1);
        for p=1:n
            seq(p,1)=min(store(p,:));
        end
        feature_depend(1,j)=sum(seq)/n;
%            feature_depend(1,j)=sum(seq);
        if feature_depend(1,j)>max_depend
            max_depend=feature_depend(1,j);
            max_feature=j;
        end
    end
    %%%%%%%%%%%%%悲观去最小%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%判断终止条件%%%%%%%%%%%%%%%%%%
    terminal=max_depend-stand;
%     
%     if terminal>=0.0001%一个终止条件
%             display('woshizhongugorer');
%             feature_slct=[feature_slct feature_lft(max_feature)];%选出的特征
%              feature_lft(max_feature)=[];%去掉已经选出的特征
%             % sample_lft(max_Pos)=[];%去掉已经在正域中的样本
%             %%%%%%%%%%%%%%输出需要的依赖度值%%%%%%%%%%%%%
%             
           % valueoflabel_first=zeros(1,length(valueoflabel));
            if num_cur==0
                feature_dependfirst=feature_depend;
            end
            if num_cur==m-1
                feature_dependlast=feature_depend;
                break;%一个终止条件
            end
            TTpes{1}=[TTpes{1} terminal];
%             terminal
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     else
%         display('%%%%%%%%%%%%%%%%%');
%         feature_dependlast=feature_depend;
%         if num_cur==0
%             feature_dependfirst=feature_dependlast;
%         end
% 
%         break;
%     end
    if num_cur~=0
    if terminal<over%一个终止条件
        feature_dependlast=feature_depend;
        if num_cur==0
            feature_dependfirst=feature_dependlast;
        end

        break;    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else
        
        feature_slct=[feature_slct feature_lft(max_feature)];%选出的特征
             feature_lft(max_feature)=[];%去掉已经选出的特征
            % sample_lft(max_Pos)=[];%去掉已经在正域中的样本
            %%%%%%%%%%%%%%输出需要的依赖度值%%%%%%%%%%%%%
            
           % valueoflabel_first=zeros(1,length(valueoflabel));
            if num_cur==0
                feature_dependfirst=feature_depend;
            end
            if num_cur==m-1
                feature_dependlast=feature_depend;
                break;%一个终止条件
            end
            
    end
    else
        feature_slct=[feature_slct feature_lft(max_feature)];
        feature_lft(max_feature)=[];
        feature_dependfirst=feature_depend;
        
    end

       stand=max_depend;   
       num_cur=num_cur+1;
    end
end
end


function [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,Num,Smooth)
%MLKNN_train trains a multi-label k-nearest neighbor classifier
%
%    Syntax
%
%       [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,num_neighbor)
%
%    Description
%
%       KNNML_train takes,
%           train_data   - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target - A QxM array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           Num          - Number of neighbors used in the k-nearest neighbor algorithm
%           Smooth       - Smoothing parameter
%      and returns,
%           Prior        - A Qx1 array, for the ith class Ci, the prior probability of P(Ci) is stored in Prior(i,1)
%           PriorN       - A Qx1 array, for the ith class Ci, the prior probability of P(~Ci) is stored in PriorN(i,1)
%           Cond         - A Qx(Num+1) array, for the ith class Ci, the probability of P(k|Ci) (0<=k<=Num) i.e. k nearest neighbors of an instance in Ci will belong to Ci , is stored in Cond(i,k+1)
%           CondN        - A Qx(Num+1) array, for the ith class Ci, the probability of P(k|~Ci) (0<=k<=Num) i.e. k nearest neighbors of an instance not in Ci will belong to Ci, is stored in CondN(i,k+1)

    [num_class,num_training]=size(train_target);

%Computing distance between training instances
    dist_matrix=diag(realmax*ones(1,num_training));
    for i=1:num_training-1
        if(mod(i,100)==0)
            disp(strcat('computing distance for instance:',num2str(i)));
        end
        vector1=train_data(i,:);
        for j=i+1:num_training            
            vector2=train_data(j,:);
            dist_matrix(i,j)=sqrt(sum((vector1-vector2).^2));
            dist_matrix(j,i)=dist_matrix(i,j);
        end
    end
    
%Computing Prior and PriorN
    for i=1:num_class
        temp_Ci=sum(train_target(i,:)==ones(1,num_training));
        Prior(i,1)=(Smooth+temp_Ci)/(Smooth*2+num_training);
        PriorN(i,1)=1-Prior(i,1);
    end

%Computing Cond and CondN
    Neighbors=cell(num_training,1); %Neighbors{i,1} stores the Num neighbors of the ith training instance
    for i=1:num_training
        [temp,index]=sort(dist_matrix(i,:));
        Neighbors{i,1}=index(1:Num);
    end
    
    temp_Ci=zeros(num_class,Num+1); %The number of instances belong to the ith class which have k nearest neighbors in Ci is stored in temp_Ci(i,k+1)
    temp_NCi=zeros(num_class,Num+1); %The number of instances not belong to the ith class which have k nearest neighbors in Ci is stored in temp_NCi(i,k+1)
    for i=1:num_training
        temp=zeros(1,num_class); %The number of the Num nearest neighbors of the ith instance which belong to the jth instance is stored in temp(1,j)
        neighbor_labels=[];
        for j=1:Num
            neighbor_labels=[neighbor_labels,train_target(:,Neighbors{i,1}(j))];
        end
        for j=1:num_class
            temp(1,j)=sum(neighbor_labels(j,:)==ones(1,Num));
        end
        for j=1:num_class
            if(train_target(j,i)==1)
                temp_Ci(j,temp(j)+1)=temp_Ci(j,temp(j)+1)+1;
            else
                temp_NCi(j,temp(j)+1)=temp_NCi(j,temp(j)+1)+1;
            end
        end
    end
    for i=1:num_class
        temp1=sum(temp_Ci(i,:));
        temp2=sum(temp_NCi(i,:));
        for j=1:Num+1
            Cond(i,j)=(Smooth+temp_Ci(i,j))/(Smooth*(Num+1)+temp1);
            CondN(i,j)=(Smooth+temp_NCi(i,j))/(Smooth*(Num+1)+temp2);
        end
    end 
end

function Average_Precision=MLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN)
%MLKNN_test tests a multi-label k-nearest neighbor classifier.
%
%    Syntax
%
%       [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=MLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN)
%
%    Description
%
%       KNNML_test takes,
%           train_data       - An M1xN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target     - A QxM1 array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           test_data        - An M2xN array, the ith instance of testing instance is stored in test_data(i,:)
%           test_target      - A QxM2 array, if the ith testing instance belongs to the jth class, test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%           Num              - Number of neighbors used in the k-nearest neighbor algorithm
%           Prior            - A Qx1 array, for the ith class Ci, the prior probability of P(Ci) is stored in Prior(i,1)
%           PriorN           - A Qx1 array, for the ith class Ci, the prior probability of P(~Ci) is stored in PriorN(i,1)
%           Cond             - A Qx(Num+1) array, for the ith class Ci, the probability of P(k|Ci) (0<=k<=Num) i.e. k nearest neighbors of an instance in Ci will belong to Ci , is stored in Cond(i,k+1)
%           CondN            - A Qx(Num+1) array, for the ith class Ci, the probability of P(k|~Ci) (0<=k<=Num) i.e. k nearest neighbors of an instance not in Ci will belong to Ci, is stored in CondN(i,k+1)
%      and returns,
%           HammingLoss      - The hamming loss on testing data
%           RankingLoss      - The ranking loss on testing data
%           OneError         - The one-error on testing data as
%           Coverage         - The coverage on testing data as
%           Average_Precision- The average precision on testing data
%           Outputs          - A QxM2 array, the probability of the ith testing instance belonging to the jCth class is stored in Outputs(j,i)
%           Pre_Labels       - A QxM2 array, if the ith testing instance belongs to the jth class, then Pre_Labels(j,i) is +1, otherwise Pre_Labels(j,i) is -1

    [num_class,num_training]=size(train_target);
    [num_class,num_testing]=size(test_target);
    
%Computing distances between training instances and testing instances
    dist_matrix=zeros(num_testing,num_training);
    for i=1:num_testing
        if(mod(i,100)==0)
            disp(strcat('computing distance for instance:',num2str(i)));
        end
        vector1=test_data(i,:);
        for j=1:num_training
            vector2=train_data(j,:);
            dist_matrix(i,j)=sqrt(sum((vector1-vector2).^2));
        end
    end

%Find neighbors of each testing instance
    Neighbors=cell(num_testing,1); %Neighbors{i,1} stores the Num neighbors of the ith testing instance
    for i=1:num_testing
        [temp,index]=sort(dist_matrix(i,:));
        Neighbors{i,1}=index(1:Num);
    end
    
%Computing Outputs
    Outputs=zeros(num_class,num_testing);
    for i=1:num_testing
%         if(mod(i,100)==0)
%             disp(strcat('computing outputs for instance:',num2str(i)));
%         end
        temp=zeros(1,num_class); %The number of the Num nearest neighbors of the ith instance which belong to the jth instance is stored in temp(1,j)
        neighbor_labels=[];
        for j=1:Num
            neighbor_labels=[neighbor_labels,train_target(:,Neighbors{i,1}(j))];
        end
        for j=1:num_class
            temp(1,j)=sum(neighbor_labels(j,:)==ones(1,Num));
        end
        for j=1:num_class
            Prob_in=Prior(j)*Cond(j,temp(1,j)+1);
            Prob_out=PriorN(j)*CondN(j,temp(1,j)+1);
            if(Prob_in+Prob_out==0)
                Outputs(j,i)=Prior(j);
            else
                Outputs(j,i)=Prob_in/(Prob_in+Prob_out);      
            end
        end
    end
    
%Evaluation
    Pre_Labels=zeros(num_class,num_testing);
    for i=1:num_testing
        for j=1:num_class
            if(Outputs(j,i)>=0.5)
                Pre_Labels(j,i)=1;
            else
                Pre_Labels(j,i)=-1;
            end
        end
    end
    Average_Precision=Average_precision(Outputs,test_target);
end




   



    