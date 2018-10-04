clear
load('dwn_big.mat');
ll = P.alpha2(169:2*168,1);

aHigh = [1.05 1.02];
aLow = [0.7 0.5];
aWeekend = [0.6 -0.5 0.4];
llHat = zeros(10*168,1);
arTermsSize = length(aHigh);
for iHour = 1:168
    rng(iHour)
    offset = rand(1);
    currentDay = floor(iHour/24);
    currentHour = mod(iHour, 24);
    if(currentDay > 4)
        arTermsSize = length(aWeekend);
        if(currentHour == 1)
            llHat(iHour, 1) = ll(iHour, 1);
        else
            llHat(iHour, 1) = 0;
            iArTerm = 1; 
            aCurrent = aWeekend;
            while( iArTerm < arTermsSize)
                 aCurrent(iArTerm)
                if( currentHour == 0)
                    currentHour = 24;
                    llHat(iHour, 1) = llHat(iHour, 1) + aCurrent(iArTerm)*llHat(currentHour - 1) + 0.3*offset;
                else
                    llHat(iHour, 1) = llHat(iHour, 1) + aCurrent(iArTerm)*llHat(currentHour - 1) + 0.3*offset;
                end
                iArTerm = iArTerm + 1;
            end
        end
    else
        if(currentHour == 1)
            llHat(iHour, 1) = ll(iHour, 1);
        else
            llHat(iHour, 1) = 0;
            iArTerm = 1;
            while( iArTerm < arTermsSize)
                if( currentHour < 18 && currentHour > 10)
                    aCurrent = aHigh;
                else
                    aCurrent = aLow;
                end
                if( currentHour == 0)
                    currentHour = 24;
                    llHat(iHour, 1) = llHat(iHour, 1) + aCurrent(iArTerm)*llHat(currentHour - 1) + 0.2*offset;
                else
                    llHat(iHour, 1) = llHat(iHour, 1) + aCurrent(iArTerm)*llHat(currentHour - 1) + 0.2*offset;
                end
                iArTerm = iArTerm + 1;
            end
        end
    end
end
figure(1)
stairs(llHat);

numSample = 1000;
rng(1);
priceError = zeros(P.Hp, numSample);
for iHorz = 1: P.Hp-1
    priceError(iHorz, :) = 1.05^iHorz*0.1*randn(1, numSample);
    negIdx = find(priceError(iHorz, :) < 0);
    %priceError(iHorz, negIdx) = zeros(1, length(negIdx));
end

% error 
startSim = 4875;

figure(2)
Y = llHat(25:96, 1);
Y = Y + 0.05*randn(size(Y));
Y(31)=1.3;
Y = max(Y,0.1);

hold on;
for iCount = 1:100
    forecastPrice = llHat( 97:97 + P.Hp -1, 1) +...
        [0;priceError( 1:P.Hp-1, iCount)];
    if any(forecastPrice)
        priceId = find( forecastPrice < 0);
        forecastPrice(priceId, 1) = 0.3*rand(length(priceId), 1);
    end 
    stairs([startSim: startSim + P.Hp], [Y(49); forecastPrice], 'Color', [0.75 0.79 0.75]);
    hold on;
end
stairs(startSim - 48 : startSim+1, Y(1:50), 'b','LineWidth', 2);
stairs(startSim +1 : startSim + P.Hp-1, Y(49+1:end), 'r','LineWidth', 2);
ylabel('electricity prices [eu/kWh]',  'FontSize', 12);
xlabel('Time [hr] ', 'FontSize', 12);
title('Electricity price prediction model');
grid on;
axis tight;