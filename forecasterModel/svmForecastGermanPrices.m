%%
clear;
load('dwn_big.mat');
pathToXlsFile = 'forecaster/germany2016.xls';
[priceXlsData, txt, raw] = xlsread(pathToXlsFile, 1); 
priceDataGermany = reshape( flip(priceXlsData(1:366, 2:25), 1)', 24*366, 1);
nanIds = find( isnan(priceDataGermany) == 1);
for iNan = 1:length(nanIds)
    priceDataGermany(nanIds(iNan)) = priceDataGermany(nanIds(iNan)-1);
end
stairs( priceDataGermany(1680:1680 + 2*24) );
axis tight;
%
initialSample = 3250;
testingSample = 1500;
trainingSamples = initialSample -1500;
svmOption.day = 0;
svmOption.hr = 0;
svmOption.predictHor = 24;
svmOption.totalData = 24*366;
svmOption.sizeTraingingData = initialSample;
svmOption.numFeatureTs = 24;

[outputPred, featureElectricityPrice, ~] = prepareSvmData(priceDataGermany, svmOption);
[ svmModelElectricPrice, svmModelFeature, svmDetails] = scaleFeatureData( outputPred(1:initialSample),...
    featureElectricityPrice(1:initialSample, :), 1 );

%% svm train for the model 
%{
optsNuSVR = '-s 4 -t 2 -c 0.3 -b 1 -v 2 -n 0.1';
%optsElsionSvm = '-s 3 -t 2  -c 0.1 -b 1 -v 2 -p 0.2';

electricityPriceSvmNuModel = svmtrain(svmelecticyPrice, featureElectricityPrice, optsNuSVR);
%}
numModels = 100;
parameterSvm.validError = zeros(1, numModels);
parameterSvm.epsilonC = zeros(1, numModels);
parameterSvm.accuracy = zeros(3, numModels);
for iCount = 1:numModels
    optsElsionSvm = '-s 3 -v 3 -h 0 -p 0.15 -c 00.100';
    parameterSvm.epsilonC(1, iCount) = 1 + 0.9*iCount;
    iOpts = num2str(parameterSvm.epsilonC(1, iCount));
    if length(iOpts) < 6
        for i = 1: 6 - length(iOpts)
           optsElsionSvm(end-length(iOpts)+i) = '0'; 
        end
        for i = 1: length(iOpts)
            optsElsionSvm(end-6 + i) = iOpts(i);
        end
    else 
        optsElsionSvm(end-5:end) = iOpts(1:6);
    end
    % traingin and cross validation data sets 
    parameterSvm.validError(1, iCount) = svmtrain( svmModelElectricPrice( 1:trainingSamples ), svmModelFeature(1:trainingSamples, :), optsElsionSvm);
    % testing sets
    finalSvmEpsilon = '-s 3 -h 0 -p 0.15 -c 00.100';
    costC = parameterSvm.epsilonC(1, iCount);
    iOpts = num2str(costC);
    if length(iOpts) < 6
        for i = 1: 6 - length(iOpts)
            finalSvmEpsilon(end-length(iOpts)+i) = '0';
        end
        for i = 1: length(iOpts)
            finalSvmEpsilon(end-6 + i) = iOpts(i);
        end
    else
        finalSvmEpsilon(end-5:end) = iOpts(1:6);
    end
    electricityPriceSvmEpsilonModel = svmtrain( svmModelElectricPrice( 1:trainingSamples ), svmModelFeature( 1:trainingSamples, :), finalSvmEpsilon);
    [svmElectricPricePredict, accuracy, ~] = svmpredict(svmModelElectricPrice( trainingSamples + 1: initialSample ),...
        svmModelFeature(trainingSamples + 1: initialSample , :), electricityPriceSvmEpsilonModel);
    parameterSvm.accuracy(:, iCount)  = accuracy;
    if(iCount == numModels)
        costC = parameterSvm.epsilonC( 1, find( parameterSvm.accuracy(3, :) == max( parameterSvm.accuracy(3, :) )));
        finalSvmEpsilon = '-s 3 -h 0 -p 0.25 -c 00.100';
        iOpts = num2str(costC);
        if length(iOpts) < 6
            for i = 1: 6 - length(iOpts)
                finalSvmEpsilon(end-length(iOpts)+i) = '0';
            end
            for i = 1: length(iOpts)
                finalSvmEpsilon(end-6 + i) = iOpts(i);
            end
        else
            finalSvmEpsilon(end-5:end) = iOpts(1:6);
        end
        electricityPriceSvmEpsilonModel = svmtrain( svmModelElectricPrice( 1:trainingSamples ), svmModelFeature( 1:trainingSamples, :), finalSvmEpsilon);
        [svmElectricPricePredict, accuracy, ~] = svmpredict(svmModelElectricPrice( trainingSamples + 1: initialSample ),...
            svmModelFeature(trainingSamples + 1: initialSample , :), electricityPriceSvmEpsilonModel);
        errorElecticPrice = svmElectricPricePredict - svmModelElectricPrice( trainingSamples + 1: initialSample );
    end
end
%% Prediction 

currentSample = initialSample; 
currentPrice = outputPred( currentSample + 1, 1);
svmOption.featureElectricityPrice = featureElectricityPrice(1:initialSample , :);

priceForecast = zeros( svmOption.predictHor, testingSample);
priceForecastError = zeros( svmOption.predictHor, testingSample);
svmOption.predictHor = 24;
svmOption.minFeature = svmDetails.minFeature;
svmOption.maxFeature = svmDetails.maxFeature;
for iSample = 1:testingSample 
    currentPrice = outputPred( trainingSamples + iSample, 1);
    svmOption.featureElectricityPrice = featureElectricityPrice(1:initialSample - testingSample + iSample,:);
    %svmOption.ll = svmModelFeature( trainingSamples + iSample + 1 , :);
    [ySvmPredict, detailsSvm]= svmPredictPrice( electricityPriceSvmEpsilonModel, currentPrice, svmOption);
    %llnorm(iSample, 1) = norm(detailsSvm.scaledCurrentFeature - svmOption.ll); 
    priceForecast(:, iSample) = ySvmPredict;
    priceForecastError(:, iSample) = outputPred( trainingSamples + iSample : trainingSamples + iSample + svmOption.predictHor - 1, 1) - priceForecast(:, iSample);  
end
%%
figure
stairs( outputPred( trainingSamples + 1:trainingSamples + P.Hp, 1), 'LineWidth', 2);
hold all;
for iCount = 1:100
    ll = priceForecast(:, 1) + priceForecastError(:,iCount);
    stairs( ll, 'Color', [0.85 0.85 0.85] );
    hold all;
end 

%%
figure 
stairs(ySvmPredict);
hold all;
stairs(outputPred( initialSample + 1:initialSample + 24, 1))
%%
%{
stairs(svmModelElectricPrice (trainingSamples +1*168+1: trainingSamples + 8*168));
hold all;
stairs(svmElectricPricePredict (1*168+1:8*168));
grid on
%electricityPriceSvmEspsilonModel = svmtrain(svmelecticyPrice, featureElectricityPrice, '-s 3 -p 0.2');

plot(yPred(1:168))
hold all;
plot(outputPred(1:168))
%}
%%
forecastError = zeros(simSamples, P.Hp);
for iSim = initialSample: initialSample + simSamples
    Q.Nt = iSim - 200;
    yPred = svmforecast(featureElectricityPrice, outputPred, svm_prediction_model, Q, sc_prms, yScale);
    forecastError(iSim - initialSample + 1 , :) = 3600*yPred - DemandData( iSim + 1: iSim + P.Hp, 1)';
    %forecastError(iSim , :) = kron(ratioDemand, errorPred)';
end
