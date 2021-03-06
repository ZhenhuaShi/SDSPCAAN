clc; clearvars; close all; rng(0);
nRepeats=8;
lambdas=10.^(-2:1:2);
ks=10:10:100;
nn=1; % NumNeighbors

nIt=length(ks);
LN={'Baseline','PCA','PCA-LDA','JPCDA','SDSPCA','JSDSPCDA','SLNP','SPCAN','SDSPCA-LPP','SDSPCAN','JSDSPCDAN'};
nAlgs=length(LN);

datasets={'Musk1','MSRA25_uni','PalmData25_uni','uspst_uni','isolet','Yale_32x32','ORL_32x32','COIL20','YaleB_32x32'};
datasets=datasets(1)

% Display results in parallel computing
dqWorker = parallel.pool.DataQueue; afterEach(dqWorker, @(data) fprintf('%d-%d ', data{1},data{2})); % print progress of parfor

[BCAtrain,BCAtest,BCAtune,BestLambda]=deal(cellfun(@(u)nan(length(datasets),nAlgs,nIt),cell(nRepeats,1),'UniformOutput',false));
[times]=deal(cellfun(@(u)nan(length(datasets),nAlgs),cell(nRepeats,1),'UniformOutput',false));
delete(gcp('nocreate'))
parpool(nRepeats);
parfor r=1:nRepeats
    dataDisp=cell(1,2);    dataDisp{1}=r; %warning off all;
    for s=1:length(datasets)
        dataDisp{2} = s;   send(dqWorker,dataDisp); % Display progress in parfor
        
        temp=load(['./' datasets{s} '.mat']);
        if isfield(temp,'data')
            data=temp.data;
            X=data(:,1:end-1); Y=data(:,end);
        elseif isfield(temp,'fea')
            X=temp.fea; Y=temp.gnd;
        else
            X=temp.X; Y=temp.Y;
        end
        uniqueY=unique(Y);
        X = zscore(X); [N0,M]=size(X);
        N=round(N0*.6);
        
        idsTrain=datasample(1:N0,N,'replace',false);
        trainX=X(idsTrain,:); trainY=Y(idsTrain);
        testX=X; testX(idsTrain,:)=[];
        testY=Y; testY(idsTrain)=[];
        % validation data
        N1=round(N0*.2);
        idsTune=datasample(1:(N0-N),N1,'replace',false);
        tuneX=testX(idsTune,:); tuneY=testY(idsTune);
        testX(idsTune,:)=[]; testY(idsTune)=[];
        idsTest=1:N0;idsTest([idsTrain idsTune])=[];
        trainInd=idsTrain;
        testInd=1:N0;testInd(idsTrain)=[];
        valInd=testInd(idsTune);
        testInd(idsTune)=[];
        MtrainX=mean(trainX);
        trainX=trainX-MtrainX; tuneX=tuneX-MtrainX; testX=testX-MtrainX;
        
        Ks=ks;
        Ks(Ks>min(size(trainX)))=[];
        if isempty(Ks)
            Ks=min(size(trainX))-1;
        end
        Ks(Ks<length(unique(trainY))-1)=[];
        if isempty(Ks)
            Ks=length(unique(trainY))-1;
        end
        maxK=max(Ks);
        
        
        %% Baseline
        tic
        id=1;
        W = eye(size(trainX,2),maxK);
        [trainXW,tuneXW,testXW] = deal(trainX*W,tuneX*W,testX*W);
        for i=1:length(Ks)
            k=Ks(i);
            model = fitcknn(trainXW(:,1:k),trainY,'NumNeighbors',nn,'Distance','euclidean','Standardize',1);
            BCAtrain{r}(s,id,i) = CacluateBCA(trainY,predict(model,trainXW(:,1:k)));
            BCAtune{r}(s,id,i) = CacluateBCA(tuneY,predict(model,tuneXW(:,1:k)));
            BCAtest{r}(s,id,i) = CacluateBCA(testY,predict(model,testXW(:,1:k)));
        end
        times{r}(s,id)=toc;
        
        
        %% PCA
        tic
        id=id+1;
        W= PCA(trainX,maxK);
        [trainXW,tuneXW,testXW] = deal(trainX*W,tuneX*W,testX*W);
        for i=1:length(Ks)
            k=Ks(i);
            model = fitcknn(trainXW(:,1:k),trainY,'NumNeighbors',nn,'Distance','euclidean','Standardize',1);
            BCAtrain{r}(s,id,i) = CacluateBCA(trainY,predict(model,trainXW(:,1:k)));
            BCAtune{r}(s,id,i) = CacluateBCA(tuneY,predict(model,tuneXW(:,1:k)));
            BCAtest{r}(s,id,i) = CacluateBCA(testY,predict(model,testXW(:,1:k)));
        end
        times{r}(s,id)=toc;
        
        
        %% PCA-LDA
        tic
        id=id+1;
        XTX=trainX'*trainX;
        OY = double(bsxfun(@eq, trainY(:), unique(trainY)'));
        OY = OY - mean(OY);
        for i=1:length(Ks)
            k=Ks(i);
            thre=-inf;
            for lambda=lambdas
                W= PCA(trainX,k);
                V=(W'*XTX*W+lambda*eye(k))\(W'*trainX'*OY);
                W=W*V;
                [trainXW,tuneXW,testXW] = deal(trainX*W,tuneX*W,testX*W);
                model = fitcknn(trainXW,trainY,'NumNeighbors',nn,'Distance','euclidean','Standardize',1);
                tmp1 = CacluateBCA(trainY,predict(model,trainXW));
                tmp2 = CacluateBCA(tuneY,predict(model,tuneXW));
                tmp3 = CacluateBCA(testY,predict(model,testXW));
                if tmp2>thre
                    thre=tmp2;
                    BestLambda{r}(s,id,i)=lambda;
                    [BCAtrain{r}(s,id,i),BCAtune{r}(s,id,i),BCAtest{r}(s,id,i)]=deal(tmp1,tmp2,tmp3);
                end
            end
        end
        times{r}(s,id)=toc;
        
        
        %% JPCDA
        tic
        id=id+1;
        for i=1:length(Ks)
            k=Ks(i);
            thre=-inf;
            for lambda=lambdas
                W= JPCDA(trainX, trainY, k, lambda);
                [trainXW,tuneXW,testXW] = deal(trainX*W,tuneX*W,testX*W);
                model = fitcknn(trainXW,trainY,'NumNeighbors',nn,'Distance','euclidean','Standardize',1);
                tmp1 = CacluateBCA(trainY,predict(model,trainXW));
                tmp2 = CacluateBCA(tuneY,predict(model,tuneXW));
                tmp3 = CacluateBCA(testY,predict(model,testXW));
                if tmp2>thre
                    thre=tmp2;
                    BestLambda{r}(s,id,i)=lambda;
                    [BCAtrain{r}(s,id,i),BCAtune{r}(s,id,i),BCAtest{r}(s,id,i)]=deal(tmp1,tmp2,tmp3);
                end
            end
        end
        times{r}(s,id)=toc;
        
        
        %% SDSPCA
        tic
        id=id+1;
        for i=1:length(Ks)
            k=Ks(i);
            W= SDSPCA(trainX,trainY,k);
            [trainXW,tuneXW,testXW] = deal(trainX*W,tuneX*W,testX*W);
            model = fitcknn(trainXW,trainY,'NumNeighbors',nn,'Distance','euclidean','Standardize',1);
            BCAtrain{r}(s,id,i) = CacluateBCA(trainY,predict(model,trainXW));
            BCAtune{r}(s,id,i) = CacluateBCA(tuneY,predict(model,tuneXW));
            BCAtest{r}(s,id,i) = CacluateBCA(testY,predict(model,testXW));
        end
        times{r}(s,id)=toc;
        
        
        %% JSDSPCDA
        tic
        id=id+1;
        for i=1:length(Ks)
            k=Ks(i);
            thre=-inf;
            for lambda=lambdas
                W= JSDSPCDA(trainX, trainY, k, [1,1,lambda]);
                [trainXW,tuneXW,testXW] = deal(trainX*W,tuneX*W,testX*W);
                model = fitcknn(trainXW,trainY,'NumNeighbors',nn,'Distance','euclidean','Standardize',1);
                tmp1 = CacluateBCA(trainY,predict(model,trainXW));
                tmp2 = CacluateBCA(tuneY,predict(model,tuneXW));
                tmp3 = CacluateBCA(testY,predict(model,testXW));
                if tmp2>thre
                    thre=tmp2;
                    BestLambda{r}(s,id,i)=lambda;
                    [BCAtrain{r}(s,id,i),BCAtune{r}(s,id,i),BCAtest{r}(s,id,i)]=deal(tmp1,tmp2,tmp3);
                end
            end
        end
        times{r}(s,id)=toc;
        
        
        %% SLNP
        tic
        id=id+1;
        tY=tabulate(trainY);
        nC=tY(:,2);
        if min(nC)-2>=2
            for i=1:length(Ks)
                k=Ks(i);
                W= SLNP(trainX,trainY,k);
                [trainXW,tuneXW,testXW] = deal(trainX*W,tuneX*W,testX*W);
                model = fitcknn(trainXW,trainY,'NumNeighbors',nn,'Distance','euclidean','Standardize',1);
                BCAtrain{r}(s,id,i) = CacluateBCA(trainY,predict(model,trainXW));
                BCAtune{r}(s,id,i) = CacluateBCA(tuneY,predict(model,tuneXW));
                BCAtest{r}(s,id,i) = CacluateBCA(testY,predict(model,testXW));
            end
        end
        times{r}(s,id)=toc;
        
        
        %% SPCAN
        tic
        id=id+1;
        for i=1:length(Ks)
            k=Ks(i);
            W= SPCAN(trainX,trainY,k);
            [trainXW,tuneXW,testXW] = deal(trainX*W,tuneX*W,testX*W);
            model = fitcknn(trainXW,trainY,'NumNeighbors',nn,'Distance','euclidean','Standardize',1);
            BCAtrain{r}(s,id,i) = CacluateBCA(trainY,predict(model,trainXW));
            BCAtune{r}(s,id,i) = CacluateBCA(tuneY,predict(model,tuneXW));
            BCAtest{r}(s,id,i) = CacluateBCA(testY,predict(model,testXW));
        end
        times{r}(s,id)=toc;
        
        
        %% SDSPCA-LPP
        tic
        id=id+1;
        for i=1:length(Ks)
            k=Ks(i);
            thre=-inf;
            for lambda=lambdas
                W= SDSPCA_LPP(trainX, trainY, k, [1,1,lambda]);
                [trainXW,tuneXW,testXW] = deal(trainX*W,tuneX*W,testX*W);
                model = fitcknn(trainXW,trainY,'NumNeighbors',nn,'Distance','euclidean','Standardize',1);
                tmp1 = CacluateBCA(trainY,predict(model,trainXW));
                tmp2 = CacluateBCA(tuneY,predict(model,tuneXW));
                tmp3 = CacluateBCA(testY,predict(model,testXW));
                if tmp2>thre
                    thre=tmp2;
                    BestLambda{r}(s,id,i)=lambda;
                    [BCAtrain{r}(s,id,i),BCAtune{r}(s,id,i),BCAtest{r}(s,id,i)]=deal(tmp1,tmp2,tmp3);
                end
            end
        end
        times{r}(s,id)=toc;
        
        
        %% SDSPCAAN
        tic
        id=id+1;
        for i=1:length(Ks)
            k=Ks(i);
            thre=-inf;
            for lambda=lambdas
                W= SDSPCAAN(trainX, trainY, k, [1,1,lambda]);
                [trainXW,tuneXW,testXW] = deal(trainX*W,tuneX*W,testX*W);
                model = fitcknn(trainXW,trainY,'NumNeighbors',nn,'Distance','euclidean','Standardize',1);
                tmp1 = CacluateBCA(trainY,predict(model,trainXW));
                tmp2 = CacluateBCA(tuneY,predict(model,tuneXW));
                tmp3 = CacluateBCA(testY,predict(model,testXW));
                if tmp2>thre
                    thre=tmp2;
                    BestLambda{r}(s,id,i)=lambda;
                    [BCAtrain{r}(s,id,i),BCAtune{r}(s,id,i),BCAtest{r}(s,id,i)]=deal(tmp1,tmp2,tmp3);
                end
            end
        end
        times{r}(s,id)=toc;
        
        
        %% JSDSPCDAN
        tic
        id=id+1;
        for i=1:length(Ks)
            k=Ks(i);
            thre=-inf;
            for lambda=lambdas
                W= JSDSPCDAN(trainX, trainY, k, [1,1,lambda,1]);
                [trainXW,tuneXW,testXW] = deal(trainX*W,tuneX*W,testX*W);
                model = fitcknn(trainXW,trainY,'NumNeighbors',nn,'Distance','euclidean','Standardize',1);
                tmp1 = CacluateBCA(trainY,predict(model,trainXW));
                tmp2 = CacluateBCA(tuneY,predict(model,tuneXW));
                tmp3 = CacluateBCA(testY,predict(model,testXW));
                if tmp2>thre
                    thre=tmp2;
                    BestLambda{r}(s,id,i)=lambda;
                    [BCAtrain{r}(s,id,i),BCAtune{r}(s,id,i),BCAtest{r}(s,id,i)]=deal(tmp1,tmp2,tmp3);
                end
            end
        end
        times{r}(s,id)=toc;
        
        
    end
end
save('demo.mat','BCAtrain','BCAtune','BCAtest','times','BestLambda','datasets','nAlgs','LN','lambdas','nRepeats','ks');

%% Plot results
clear
load demo
totalHours=nansum(reshape(cat(1,times{:}),1,[]))/3600/8
datasetsName={'Musk1','MSRA25','Palm','USPST','Isolet','Yale','ORL','COIL20','YaleB'};
close all;
lineStyles={'k--','k-','g--','g-','b--','b-','r--','r-','m--','m-','c--','c-'};
ids=1:length(LN);
figure;
set(gcf,'DefaulttextFontName','times new roman','DefaultaxesFontName','times new roman','defaultaxesfontsize',12);
hold on;
for s=1:length(datasets)
    tmpt=cellfun(@(u)squeeze(u(s,:,:)),BCAtune,'UniformOutput',false);
    tmpt=nanmean(cat(3,tmpt{:}),3);
    for i=ids
        plot(ks,tmpt(i,:),lineStyles{i},'linewidth',2);
    end
    set(gca,'yscale','log');
    xlabel('Dimensionality'); ylabel('BCA'); box on; axis tight;
    title(datasetsName{s});
end
legend(LN(ids),'FontSize',12,'NumColumns',1,'Location','eastoutside');
[tmp,ttmp]=deal(nan(length(datasets),length(LN),nRepeats));
for s=1:length(datasets)
    ttmp0=cellfun(@(u)squeeze(u(s,ids)),times,'UniformOutput',false);
    ttmp(s,ids,:)=cat(1,ttmp0{:})';
    for id=1:length(LN)
        try
            tmp(s,id,:)=cell2mat(cellfun(@(u,m)squeeze(u(s,id,find(m(s,id,:)==max(m(s,id,:)),1))),BCAtest,BCAtune,'UniformOutput',false));
        catch
        end
    end
end
A=[nanmean(nanmean(tmp(:,ids,:),1),3);
    nanstd(nanmean(tmp(:,ids,:),1),[],3);
    nanmean(nanmean(ttmp(:,ids,:),1),3);
    nanstd(nanmean(ttmp(:,ids,:),1),[],3)];
a=squeeze(nanmean(tmp(:,ids,:),3));
a=[a;nanmean(a,1)]; sa=sort(a,2);
b=a==sa(:,1);c=a==sa(:,2);
at=squeeze(nanmean(ttmp(:,ids,:),3));
al=nanmean(cat(3,BestLambda{:}),3); al=[al;nanmean(al,1)];