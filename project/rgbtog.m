function [outputArg] = rgbtog(inputArg)
    %Converts a matrix containing for each row an RGB image into a matrix
    %containing for each row a grayscale image using function rgb2gray()

    [sz1,sz2] = size(inputArg);
    
    grayscaletrain = zeros(sz1,32,32,'uint8');

    for i = 1:sz1
        aux = rcm_img(inputArg(i,:,:,:));
        grayscaletrain(i,:,:) = rgb2gray(aux);
    end
    
    outputArg = grayscaletrain;
    
end

