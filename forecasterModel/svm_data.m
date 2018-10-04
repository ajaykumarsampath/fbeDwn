function [y,x,sc_prms,P,yscale] = svm_data(Dd,N,Nt,Np,Npr,Nday,Nhr)
%svm_data function output the demand data in time series format which is
%used for training.
% 
% Input :   Dd = input data to predict;
%            N = Length length of total data;
%           Np = Amount of the past data used to predicite the future;
%          Npr = amount of future data;
%         Nday = 0/1 use days of the week as a feature;
%          Nhr = 0/1 use the hr of the day as a feture;
%
% Output :   Y = output of vector of the SVM  
%            x = input feature vector of the data 
%      scm_prm = scaling used for each input vector (before giving to the input vector)
%            P = structre containing the details of the SVM 
%                N : total data 
%               Np : training data
%               Np : previous time series features feature
%              Npr : future prediction length 
%             Nday : use day of week as feature 
%              Nhr : use hr of the day as feature 
%            rFeat : total number of fetures
%      yscale  = output scale parameter.
%

P = struct;
P.N = N;  % total data
P.Nt = Nt; % training data 
P.Np = Np;
P.Npr = Npr;  % Future data to predict
yscale = 10/max(Dd(P.Np+1:end,1));
y = 10*Dd(P.Np+1:end,1)./max(Dd(P.Np+1:end,1));
P.Nday = Nday;
P.Nhr = Nhr;
P.day = 1;
P.hr = 1;
P.rFeat_st = 1;
P.Ndr = 6*(P.Nday>0)+23*(P.Nhr>0);
P.rFeat = P.Np+P.Ndr;
%P.rFeat=229;
x = zeros(P.N-P.Np,P.Np+P.Ndr);
for i=1:P.N-P.Np
    if(P.Nday > 0)
        if(P.day > 24)
            x(i,floor((P.day-1)/24)) = 1;
            P.day = P.day+1;
            if(P.day > 24*7)
                P.day = 1;
            end
        else
            P.day = P.day+1;
        end
    end
    if(P.Nhr > 0)
        if(P.hr > 1)
            x(i,5 + P.hr) = 1;
            if(P.hr == 24)
                P.hr = 1;
            else
                P.hr = P.hr+1;
            end
        else
            P.hr = P.hr+1;
        end
    end
    x(i,P.Ndr+1:P.Np+P.Ndr) = Dd(i:i+P.Np-1,1)';
end

% Do the scaling:
nFeat = size(x,2);
sc_prms = zeros(nFeat,2);
for i=1:nFeat,
    sc_prms(i,1) = min(x(:,i));
    sc_prms(i,2) = max(x(:,i));
    if((sc_prms(i,2)-sc_prms(i,1))>0)
        x(:,i) = (x(:,i) - sc_prms(i,1))/(sc_prms(i,2)-sc_prms(i,1));
    end
end

end

