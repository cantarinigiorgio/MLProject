function [] = holdoutCV(X,clr)
    %K-fold Cross-Validation 
    %K = 4 ; split Input (12000 images) into 70% (Training)(9000 images) and
    %30%(Validation)(3000 images) four times taking different folds to do Cross-Validation
    
    
    %Validation
    X1val = vertcat(X(1:1500,:),X(6001:7500,:));  
    X2val = vertcat(X(1501:3000,:),X(7501:9000,:)); 
    X3val = vertcat(X(3001:4500,:),X(9001:10500,:));  
    X4val = vertcat(X(4501:6000,:),X(10501:12000,:));  
    
    %Train
    X1train = vertcat(X(1501:6000,:),X(7501:12000,:));    
    X2train = vertcat(X(1:1500,:),X(3001:6000,:),X(6001:7500,:),X(9001:12000,:));  
    X3train = vertcat(X(1:3000,:),X(4501:6000,:),X(6001:9000,:),X(10501:12000,:)); 
    X4train = vertcat(X(1:4500,:),X(6001:10500,:)); 

    %Converts from unit8 to double
    X1val = double(X1val);  
    X2val = double(X2val); 
    X3val = double(X3val);   
    X4val = double(X4val);
    X1train = double(X1train);  
    X2train = double(X2train); 
    X3train = double(X3train);   
    X4train = double(X4train);
    
    
    if (clr == "gs")
        [featureExtractedgs,errorgs] = extractFeatures_applySVMgrayscale(X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val);
    end
    
    if (clr == "rgb")
        [featureExtractedrgb,errorrgb] = extractFeatures_applySVMRGB(X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val);
    end
    
    if (clr == "lab")
        [featureExtractedlab,errorlab] = extractFeatures_applySVMLAB(X1train,X1val,X2train,X2val,X3train,X3val,X4train,X4val);
    end
    
    
    %%%Plot error for each extracted feature
    
    
    
    %%Grayscale
    
    error = errorgs;
    featureExtracted = featureExtractedgs;
    
    figure
    plot(0:length(error)-1, error)
    title("Grayscale");
    txt = sprintf(' H8 = HOG8 \n H4 = HOG4 \n H4E = COMBINE EDGES-HOG4 \n H16 = HOG16 \n H8E = COMBINE EDGES-HOG8 \n E = EDGES \n STDH8 = COMBINE LocalStdDeviation-HOG8 \n STD = LocalStdDeviation \n RNG = LocalRange \n LBP8 = Local Binary Pattern');

    legend(txt,'Location','southeast');
    yticks(0:.02:.5)
    xticks([0 1 2 3 4 5 6 7 8 9 10])
    xticklabels(featureExtracted)
    xlabel('featureExtracted')
    ylabel('error(%)')
    x0=30;
    y0=30;
    % xlim([0 17])
    ylim([0 0.5])
    width=800;
    height=450;
    set(gcf,'units','points','position',[x0,y0,width,height])



    %%RGB

    error = errorrgb;
    featureExtracted = featureExtractedrgb;

    figure
    plot(0:length(error)-1, error)
    txt1 = sprintf(' H8 = HOG8 \n H4 = HOG4 \n H16 = HOG16 \n E = EDGES \n RH4 = COMBINE LocalRange-HOG4 \n RH8 = COMBINE LocalRange-HOG8 \n RNG = LocalRange \n STD = LocalStdDeviation');

    legend(txt1,'Location','southeast');
    title("RGB");
    yticks(0:.02:.5)
    xticks([0 1 2 3 4 5 6 7 8])
    xticklabels(featureExtracted)
    xlabel('featureExtracted')
    ylabel('error(%)')
    x0=30;
    y0=30;
    % xlim([0 17])
    ylim([0 0.5])
    width=800;
    height=450;
    set(gcf,'units','points','position',[x0,y0,width,height])


    
    %%%LAB
    
    error = errorlab;
    featureExtracted = featureExtractedlab;

    figure
    plot(0:length(error)-1, error)
    title("LAB");
    txt2 = sprintf(' H8 = HOG8 \n H4 = HOG4 \n H16 = HOG16 \n STDH4 = COMBINE LocalStdDeviation-HOG4 \n STDH8 = COMBINE LocalStdDeviation-HOG8 \n RH4 = COMBINE LocalRange-HOG4 \n RH8 = COMBINE LocalRange-HOG8 \n RNG = LocalRange \n STD = LocalStdDeviation \n E = EDGES');
    legend(txt2,'Location','southeast');
    yticks(0:.02:.5)
    xticks([0 1 2 3 4 5 6 7 8 9 10])
    xticklabels(featureExtracted)
    xlabel('featureExtracted')
    ylabel('error(%)')
    x0=30;
    y0=30;
    % xlim([0 17])
    ylim([0 0.5])
    width=800;
    height=450;
    set(gcf,'units','points','position',[x0,y0,width,height])
      
    
end

