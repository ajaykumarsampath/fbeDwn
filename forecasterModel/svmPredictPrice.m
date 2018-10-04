function [ ySvmPredict, details] = svmPredictPrice( svmModel, currentPrice, svmOption)
%function [ yPredict] = svmPredictPrice( svmModel, x,y,P,sc_prms,yscale)
% svmforcast function forcastes the demand using the svm_model 
% 
%  input :  svm_model  =  svm model used to extimate the parameters
%                   x  =  input feature vector for the svm model
%                   y  =  output feature vector of the svm model
%                   P  =  svm prediction details 
%             sc_prms  =  feature vector scaling parameters 
%              yscale  =  output vector scaling. 
% output  :  
%                  yp  =  output vector predicted for the 24 hrs 

ySvmPredict = zeros(svmOption.predictHor, 1);
svmFeature = svmOption.featureElectricityPrice;
calenderFeature = 7*(svmOption.day > 0) + 24*(svmOption.hr > 0);
%historicFeature = svmOption.numFeatureTs; 

%minFeature = min(svmFeature);
%maxFeature = max(svmFeature);
minFeature = svmOption.minFeature;
maxFeature = svmOption.maxFeature;
rangeFeature = maxFeature - minFeature;
dimFeature = size( svmFeature );
lastFeature = svmFeature( dimFeature(1), :);

%pastWeek = svmFeature( dimFeature(1) - 24 + 1: dimFeature(1), 1:7);

pastDay = find( svmFeature(dimFeature(1), 1:7) == 1);
pastHour = sum( svmFeature( dimFeature(1) - 24 + 1: dimFeature(1), pastDay) );
currentFeature = zeros( svmOption.predictHor , dimFeature(2));
scaledCurrentFeature = zeros( svmOption.predictHor , dimFeature(2));

currentHour = pastHour;
currentDay = pastDay;
currentFeature(1, :) = lastFeature;
nSamples = 1;
for iSample = 1:nSamples
    for iCount = 1:svmOption.predictHor
        %{
        if( currentHour == 24 )
            currentDay = currentDay + 1;
            currentHour = 8;
            currentFeature(iCount, currentDay ) = 1;
            currentFeature(iCount, currentHour ) = 1;
        else
            currentHour = currentHour + 1;
            currentFeature(iCount, currentDay ) = 1;
            currentFeature(iCount, 7 + currentHour ) = 1;
        end
        %}
        if( iCount == 1 )
            currentFeature(iCount, calenderFeature + 1: dimFeature(2) ) = [currentFeature(iCount, calenderFeature + 2:end) currentPrice];
            
            %{
            iFeatureMin = min([minFeature; currentFeature(iCount, :)]);
            iFeatureMax = max([maxFeature; currentFeature(iCount, :)]);
            iFeatureRange = iFeatureMax - iFeatureMin;
            %}
            for iFeature = 1: dimFeature(2)
                %scaledCurrentFeature( iCount, iFeature) = 2*( currentFeature(iCount, iFeature) - iFeatureMin(1, iFeature))/iFeatureRange(1, iFeature) - 1;
                scaledCurrentFeature( iCount, iFeature) = 2*( currentFeature(iCount, iFeature) - minFeature(1, iFeature))/rangeFeature(1, iFeature) - 1;
            end
        else
            currentFeature( iCount, calenderFeature + 1: dimFeature(2) ) = [currentFeature(iCount - 1, calenderFeature + 2:end) ySvmPredict(iCount - 1, iSample)];
            %{
            iFeatureMin = min([iFeatureMin; currentFeature(iCount, :)]);
            iFeatureMax = max([iFeatureMax; currentFeature(iCount, :)]);
            iFeatureRange = iFeatureMax - iFeatureMin;
            %}
            for iFeature = 1: dimFeature(2)
                %scaledCurrentFeature( iCount, iFeature) = 2*( currentFeature(iCount, iFeature) - iFeatureMin(1, iFeature))/iFeatureRange(1, iFeature) - 1;
                scaledCurrentFeature( iCount, iFeature) = 2*( currentFeature(iCount, iFeature) - minFeature(1, iFeature))/rangeFeature(1, iFeature) - 1;
            end
        end
        ySvmPredict(1:iCount, iSample) = svmpredict( ySvmPredict(1:iCount, iSample), scaledCurrentFeature(1:iCount, :), svmModel);
    end
end
details.scaledCurrentFeature = scaledCurrentFeature;
end

