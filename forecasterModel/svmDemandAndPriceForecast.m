%%
clear;

intialSample =  3250;
load('dwn_big');
%load('dwn');
P.Hp = 24;
P.Hu = 23;
P.xs = 0.35*S.xmax;
Dd = DemandData(:,1); % My data
% Define data
treeOpts.Tmax = 1500; % time steps for which data are available
treeOpts.nScen = 1400; % no. of scenarios
treeOpts.N = 23; % prediction
ERROR_FORECASTER = 0;
% Compare all demands
ratioDemand = zeros(S.nd, 1);
for i = 1:S.nd
    ratioDemand(i,1) = DemandData(1,i)./Dd(1,1);
end
DemandData = 3600*DemandData;

% modeling data for svm forecaster
[yData, fData, sc_prms, Q, yScale] = svm_data(Dd, 8760, intialSample, 200, P.Hp, 1, 1);
load('mdl/svm_predictive_3_2.mat');
load('forecastError.mat');
demandForecastError = forecastError/3600;

%
% plot the demand and price
startSim = 4875;
histSimHorison = 48;
demandId = 1;
demandIdData = DemandData( startSim - histSimHorison + 1: startSim + P.Hp -1, demandId)/3600;
figure(1)
plot(startSim - histSimHorison + 1 : startSim + P.Hp - 1, demandIdData, 'LineWidth', 2);
hold on;


% svm forecast
Q.Nt = startSim - 200;
yPred = svmforecast(fData, yData, svm_prediction_model, Q, sc_prms, yScale);
svmForecast = [DemandData( startSim, 1)/3600; yPred(1, 1:P.Hp-1)'];

%% prices

clear 
load('dwn_big');
pathToXlsFile = 'forecaster/germany2016.xls';
load('mdl/svmPricePredictor.mat');
load('priceForecastOutput.mat')
[priceXlsData, txt, raw] = xlsread(pathToXlsFile, 1); 
priceDataGermany = reshape( flip(priceXlsData(1:366, 2:25), 1)', 24*366, 1);
nanIds = find( isnan(priceDataGermany) == 1);
for iNan = 1:length(nanIds)
    priceDataGermany(nanIds(iNan)) = priceDataGermany(nanIds(iNan)-1);
end

%
startSim = 4875;
histSimHorison = 48;
initialSample = 3250;
testingSample = 1500;
trainingSamples = initialSample -1500;
svmOption.day = 1;
svmOption.hr = 1;
svmOption.predictHor = 24;
svmOption.totalData = 24*366;
svmOption.sizeTraingingData = initialSample;
svmOption.numFeatureTs = 168;

[outputPred, featureElectricityPrice, ~] = prepareSvmData(priceDataGermany, svmOption);
[ svmModelElectricPrice, svmModelFeature, svmDetails] = scaleFeatureData( outputPred(1:initialSample),...
    featureElectricityPrice(1:initialSample, :), 1 );


currentPrice = outputPred( startSim - 168, 1);
ySvmPredict = svmPredictPrice( electricityPriceSvmEpsilonModel, currentPrice, svmOption);
priceForecast = ySvmPredict;
priceForecast = [currentPrice; priceForecast(1:P.Hp-1, 1)]; 


priceSim = priceDataGermany(startSim - histSimHorison + 1: startSim + P.Hp -1, 1);
figure(1)
stairs(startSim - histSimHorison + 1 : startSim + P.Hp - 1, priceSim, 'LineWidth', 2);
hold on;

for iCount = 1:100
    forecastPrice = priceForecast + [0;priceForecastError( 1:23, iCount)];
    stairs([ startSim : startSim + P.Hp-1], forecastPrice);
    hold on;
end

%%
forecastError = zeros(simSamples, P.Hp);
for iSim = intialSample: intialSample + simSamples
    
    %forecastError(iSim , :) = kron(ratioDemand, errorPred)';
end



for iCount = 1:100
    forecastDemand = demandIdData( 49:49 + forecaster.N -1, 1) +...
        [0;YpredErr( 1:23, iCount)];
    plot([ startSim: startSim + forecaster.N-1], forecastDemand);
    hold on;
end
ylabel('Water Demand $[\mathrm{m^3/s}]$', 'interpreter','latex', 'FontSize', 14);
xlabel('Time [hr] ', 'FontSize', 14, 'interpreter','latex')
title('Demand prediction with arima prediction model');
grid on;
axis tight;

%%
fileID = fopen('outputControl/controlOutput32.json', 'r');
readA = textscan(fileID, '%s');

controlAPG = zeros(48, S.nu);
iSize = 1;
iSim = 1;
while iSize < length(readA{1})
    iElement = readA{1}{iSize};
    if(strcmp(iElement, '"control"'))
        iSize = iSize + 1;
        for iControl = 1:S.nu
            iElement = readA{1}{iSize + iControl};
            if(iControl  == 1)
                controlAPG(iSim, iControl) = str2num(iElement(2:end-1));
            else
                controlAPG(iSim, iControl) = str2num(iElement(1:end-1));
            end
        end
        iSize = iSize + S.nu + 1 + 3;
        iSim = iSim + 1;
    end
end

constnorm = zeros(1, 48);
LL =  S.E*S.E';
for iSim = 1:48
    currentDemand = DemandData(startSim, :);
    constVariation(iSim, :) = (S.E*controlAPG(iSim, :)' +S.Ed*currentDemand')/3600;
    constnorm(1, iSim) = norm((S.E*controlAPG(iSim, :)' +S.Ed*currentDemand')/3600);
    ll = LL\(-S.Ed*currentDemand' - S.E*controlAPG(iSim, :)' );
    projControl(iSim, :) = (controlAPG(iSim, :)' + S.E'*ll)';
    projControlVariation(iSim, :) = (S.E*projControl(iSim, :)' +S.Ed*currentDemand')/3600;
    projControlNorm(1, iSim) = norm((S.E*projControl(iSim, :)' +S.Ed*currentDemand')/3600);
end
%%
% plot the data

scenarioNum = [6;30;86;195;224;497;631];
% 1 - economic; 2 - smooth; 3 - saftey
kpiWithPriceUncert = zeros(length(scenarioNum), 3);
kpiWithoutPriceUncert = zeros(length(scenarioNum), 3);

kpiWithPriceUncert(1, 1) = 1663.84;
kpiWithPriceUncert(2, 1) = 1662.74;
kpiWithPriceUncert(3, 1) = 1633.4;
kpiWithPriceUncert(4, 1) = 1604.02;
kpiWithPriceUncert(5, 1) = 1706.14;
kpiWithPriceUncert(6, 1) = 1723.57;
kpiWithPriceUncert(7, 1) = 1726.22;

kpiWithPriceUncert(1, 2) = 1048.17;
kpiWithPriceUncert(2, 2) = 1092.81;
kpiWithPriceUncert(3, 2) = 1093.09;
kpiWithPriceUncert(4, 2) = 1091.03;
kpiWithPriceUncert(5, 2) = 1096.25;
kpiWithPriceUncert(6, 2) = 1099.41;
kpiWithPriceUncert(7, 2) = 1123.58;

kpiWithPriceUncert(1, 3) = 1778.17;
kpiWithPriceUncert(2, 3) = 1819;
kpiWithPriceUncert(3, 3) = 1746.46;
kpiWithPriceUncert(4, 3) = 1689.65;
kpiWithPriceUncert(5, 3) = 1622.87;
kpiWithPriceUncert(6, 3) = 1551.45;
kpiWithPriceUncert(7, 3) = 1540.38;


kpiWithoutPriceUncert(1, 1) = 1648.41;
kpiWithoutPriceUncert(2, 1) = 1656.1;
kpiWithoutPriceUncert(3, 1) = 1658.43;
kpiWithoutPriceUncert(4, 1) = 1687.81;
kpiWithoutPriceUncert(5, 1) = 1761.91;
kpiWithoutPriceUncert(6, 1) = 1737.86;
kpiWithoutPriceUncert(7, 1) = 1745.36;

kpiWithoutPriceUncert(1, 2) = 1089.75;
kpiWithoutPriceUncert(2, 2) = 1087.01;
kpiWithoutPriceUncert(3, 2) = 1089.5;
kpiWithoutPriceUncert(4, 2) = 1092.96;
kpiWithoutPriceUncert(5, 2) = 1091.74;
kpiWithoutPriceUncert(6, 2) = 1097.75;
kpiWithoutPriceUncert(7, 2) = 1125.82;

kpiWithoutPriceUncert(1, 3) = 1773.79;
kpiWithoutPriceUncert(2, 3) = 1823.09;
kpiWithoutPriceUncert(3, 3) = 1670.77;
kpiWithoutPriceUncert(4, 3) = 1624.08;
kpiWithoutPriceUncert(5, 3) = 1665.07;
kpiWithoutPriceUncert(6, 3) = 1526.43;
kpiWithoutPriceUncert(7, 3) = 1545.41;


figure(2)
[hAx1, pt1, pt2] = plotyy(scenarioNum, [kpiWithPriceUncert(:,1), kpiWithoutPriceUncert(:, 1)],...
    scenarioNum, [kpiWithPriceUncert(:, 3), kpiWithoutPriceUncert(:, 3)] );
grid on;
pt1(1).LineWidth = 1.5;
pt1(2).LineWidth = 1.5;
pt2(1).LineWidth = 1.5;
pt2(2).LineWidth = 1.5;
pt1(2).LineStyle = '--';
pt2(2).LineStyle = '--';
hAx1(1).FontSize = 14;
hAx1(2).FontSize = 14;
set(get(hAx1(1),'Ylabel'),'String', 'economical');
set(get(hAx1(2),'Ylabel'),'String', 'safety');
legend(hAx1(1),'economical with price uncertainty',' economical without price uncertainty');
legend(hAx1(2),'safety with price uncertainty','safety without price uncertainty');
%axes(hAx(1))
%axis(hAx1, [0 650 1500 1900])
%axes(hAx(2))
%axis(hAx2,[0 650 1500 1900])
xlabel('Scenarios');
%% decision variables 

scenarios = zeros(length(branchingFactor), 1);
primalVariable = zeros(length(branchingFactor), 1);
dualVariable = zeros(length(branchingFactor), 1);

for iCount = 1 : length(branchingFactor)
    scenarios(iCount, 1) = length(Tree{iCount}.leaves);
    primalVariable(iCount, 1) = (S.nx + S.nu)*length(Tree{iCount}.stage);
    dualVariable(iCount, 1) = (2*S.nx + S.nu)*length(Tree{iCount}.stage);
end 

%%
%{
iSim = 1;
Ee = S.E;
Ed = S.Ed;
u = controlAPG(iSim, :)';
u(:,2) = controlAPG(iSim, :)';
lb = sysPrcndDist.umin(1:S.nu);
ub = sysPrcndDist.umax(1:S.nu);
currentDemand = DemandData(startSim, :)';
LLd = -Ee'*(LL\Ed);
LLe =  -Ee'*(LL\Ee);
for iter = 1:500
    y = u(:,1) + LLd*currentDemand + LLe*u(:,1);
    u(:,1) = max(y, lb);
    u(:,1) = min(u(:,1), ub);
    normXy(iter) = norm(u(:,1) - y);
end
u(:, 3) = gurobiControl.controlU(:,1);
x2 = S.A*controller.currentX + 1/3600*S.B*u(:,1) + 1/3600*S.Gd*currentDemand;
x2(:, 2) = S.A*controller.currentX + 1/3600*S.B*u(:,2) + 1/3600*S.Gd*currentDemand;
x2(:, 3) = S.A*controller.currentX + 1/3600*S.B*gurobiControl.controlU(:, 1)...
    + 1/3600*S.Gd*currentDemand;
plot(normXy)
find(S.xmin > x2(:,1))
find(S.xmin > x2(:,2))
y1 = (x2(:,1) -x2(:,2));
y1(:,2) = (x2(:,1) -x2(:,3));
u1 = (u(:,1) -u(:,2));
u1(:,2) = (u(:,1) -u(:,3));
%}