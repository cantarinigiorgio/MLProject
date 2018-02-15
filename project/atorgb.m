function [outputArg] = atorgb(inputArg)
    %Converts a matrix (60000x3072) into a 4-D matrix (60000x32x32x3);
    %each row contains an RGB image
    
    [sz1,~] = size(inputArg);
    
    outputArg = zeros(sz1,32,32,3,'uint8');
    
    for i = 1:sz1
        outputArg(i,:,:,:) = imrotate(rcm_img(inputArg(i,:)),270);
    end

end

