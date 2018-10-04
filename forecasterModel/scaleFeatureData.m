function [ scaledOutput, scaledInputFeature, details] = scaleFeatureData( output, inputFeature, scaleOption )
% This functions scales the input data to the svm. This scaling can be
% 1) hard normalisation 2) soft normalisation. Hard normalisation scales
% the colomns [-1 1] and soft normalisation use yScale = (y-yMean)/yStd
% 
%  Syntax : 
%  [ ySvmScaled, scaledSvmFeatures, detailsSvm] = scaleFeatureData( ySvm, svmFeatures, scaleOptions )
% 
%   INPUT-- output            :     output of the predictor 
%           inputFeatures     :     features of the predictor
%           scaleOptions      :     either 1 or 0, 1 for hard normalisation
%                                                  0 for soft normalisation
% 
%  OUTPUT-- scaledOutput        :     scaled output
%           scaledInputFeatures :     scaled features of the predictor
%           details             :     minOutPut   - minimum features colums
%                                     maxOutput   - maximum features colums
%                                     rangeOutput - range of features colums
%                                     outputScale - maximum of the output
%
%

if(scaleOption == 1)
    details.minFeature = min( inputFeature );
    details.maxFeature = max( inputFeature );
    details.rangeFeature = details.maxFeature - details.minFeature;
    scaledInputFeature = zeros(size(inputFeature));
    for iFeature = 1: size(inputFeature, 2)
        scaledInputFeature(:, iFeature) = 2*(inputFeature(:, iFeature) - details.minFeature(1, iFeature))...
            /details.rangeFeature(1, iFeature) - 1;
    end
    
    % output
    %details.yScale = 1/max( output );
    details.yScale = 1;
    scaledOutput = 1*details.yScale*output;
    
else
    
    details.minFeature = min( inputFeature );
    details.maxFeature = max( inputFeature );
    details.rangeFeature = details.maxFeature - details.minFeature;
    scaledInputFeature = inputFeature;
    %{
    scaledInputFeature = zeros(size(inputFeature));
    for iFeature = 1: size(inputFeature, 2)
        scaledInputFeature(:, iFeature) = 2*(inputFeature(:, iFeature) - details.minFeature(1, iFeature))...
            /details.rangeFeature(1, iFeature) - 1;
    end
    %}
    % output
    %details.yScale = 1/max( output );
    scaledOutput = output;
end

end

