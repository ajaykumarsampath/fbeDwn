function [ySvm , featureSvm, details] = prepareSvmData(inputData, svmOption)
%function [ySvm, featureSvm, details ] = prepareSvmData( inputData, svmOptions)
%svm_data function output the demand data in time series format which is
%used for training.
% 
% Input :   inputData = input data to predict;
%           svmOption = options of the svm data 
%                     length of total data;
%                     length of training data; 
%                     amount of the past data used to predicite the future,
%                     length of feature vector
%                     day 0/1 use days of the week as a feature
%                     hr 0/1 use the hr of the day as a feture;
%
% Output :   ySvm = output of vector of the SVM  
%            scaledFeatureSvm = input feature vector of the data 
%            details = options after preparing the svm data
%

details = struct;
details.day = 1;
details.hr = 1;
% features related to day and hr of the week
details.featureDayHr = 7*(svmOption.day > 0) + 24*(svmOption.hr > 0);
details.totalFeature = svmOption.numFeatureTs + 7*(svmOption.day > 0) + 24*(svmOption.hr > 0);

% feature input data
numData = length(inputData) - svmOption.numFeatureTs;
featureDayHr = details.featureDayHr;
featureSvm = zeros(numData , details.totalFeature);
iHour = 1;
iDay = 1;
%{
if(details.hr > 1)
    featureSvm(iData,5 + details.hr) = 1;
    if(details.hr == 24)
        details.hr = 1;
    else
        details.hr = details.hr+1;
    end
else
    details.hr = details.hr+1;
end
if(details.day > 24)
    featureSvm(iData,floor((details.day-1)/24)) = 1;
    details.day = details.day+1;
    if(details.day > 24*7)
        details.day = 1;
    end
else
    details.day = details.day+1;
end
%}
for iData = 1:numData
    if(svmOption.hr > 0) 
        featureSvm( iData, 7 + iHour) = 1;
    end
    if(svmOption.day > 0)
        featureSvm(iData, iDay) = 1;
    end
    featureSvm(iData, featureDayHr + 1: end) = inputData(iData:iData + svmOption.numFeatureTs - 1, 1)';
    if( mod(iData, 24) == 0)
        iHour = 1;
        if(iDay > 6)
            iDay = 1;
        else
            iDay = iDay + 1;
        end 
    else
        iHour = iHour + 1;
    end
end

ySvm = inputData( svmOption.numFeatureTs + 1:end, 1);

%{
% scale the features and output from [-1 1]
details.minFeature = min( featureSvm );
details.maxFeature = max( featureSvm );
details.rangeFeature = details.maxFeature - details.minFeature;
details.featureSvm = featureSvm;
for iFeature = 1: details.totalFeature
    scaledFeatureSvm(:, iFeature) = 2*(featureSvm(:, iFeature) - details.minFeature(1, iFeature))...
        /details.rangeFeature(1, iFeature) - 1;
end

% output 
details.yScale = 1/max( inputData( svmOption.numFeatureTs + 1:end, 1) );

%}

%{
% Do the scaling:
nFeat = size(featureSvm,2);
sc_prms = zeros(nFeat,2);
for iData=1:nFeat,
    sc_prms(iData,1) = min(featureSvm(:,iData));
    sc_prms(iData,2) = max(featureSvm(:,iData));
    if((sc_prms(iData,2)-sc_prms(iData,1))>0)
        featureSvm(:,iData) = (featureSvm(:,iData) - sc_prms(iData,1))/(sc_prms(iData,2)-sc_prms(iData,1));
    end
end
%}

end

