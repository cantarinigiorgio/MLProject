function [label_names,Images,Labels] = load_cifar()

    %LOAD_CIFAR-10 dataset batches  
    %return label_names that is a vector (10x1) containing the name of the labels,
    %trainingImages that is a matrix (60000x3072) containing images  and trainingLabels
    %that is a vector (60000x1) containing the labels associated with trainingImages
    
    label_names = load('cifar-10\batches_meta');
    label_names = label_names.label_names;

	aux1 = load('cifar-10\data_batch_1');
	data1 = aux1.data;
	labels1 = aux1.labels;
    
	aux1 = load('cifar-10\data_batch_2');
	data2 = aux1.data;
	labels2 = aux1.labels;
    
	aux1 = load('cifar-10\data_batch_3');
	data3 = aux1.data;
	labels3 = aux1.labels;
    
	aux1 = load('cifar-10\data_batch_4');
	data4 = aux1.data;
	labels4 = aux1.labels;
    
	aux1 = load('cifar-10\data_batch_5');
	data5 = aux1.data;
	labels5 = aux1.labels;
    
    aux1 = load('cifar-10\test_batch');
    data6 = aux1.data;
    labels6 = aux1.labels;
     
    Images = cat(1,data1,data2,data3,data4,data5,data6);
    Labels = cat(1,labels1,labels2,labels3,labels4,labels5,labels6);
      
end

