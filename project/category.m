function [outputArg1] = category(categoryLabel,label_names,inputImgs,inputLabels)
    %Return a matrix containing all images of a specified category label

    [~,~,~,sz4] = size(inputImgs);
    
    if (sz4 == 3)
        outputArg1 = zeros(6000,32,32,3,'uint8');
        j = 1;

        for i = 1:60000
            if(label_names(inputLabels(i)+1)==categoryLabel)
                outputArg1(j,:,:,:) = inputImgs(i,:,:,:);
                j = j+1;
            end
        end
    end
    
    if (sz4 == 1)
        outputArg1 = zeros(6000,32,32,'uint8');
        j = 1;

        for i = 1:60000
            if(label_names(inputLabels(i)+1)==categoryLabel)
                outputArg1(j,:,:) = inputImgs(i,:,:);
                j = j+1;
            end
        end
    end

end

