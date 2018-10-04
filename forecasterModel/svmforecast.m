function [ yPredict] = svmforecast( x,y,svm_model,P,sc_prms,yscale)

% svmforcast function forcastes the demand using the svm_model 
% 
%  input :          x  =  input feature vector for the svm model
%                   y  =  output feature vector of the svm model 
%           svm_model  =  svm model used to extimate the parameters 
%                   P  =  svm prediction details 
%             sc_prms  =  feature vector scaling parameters 
%              yscale  =  output vector scaling. 
% output  :  
%                  yp  =  output vector predicted for the 24 hrs 

yPredict = zeros(1,P.Npr);
%pmse=zeros(1,1);
nSamples = 1;
for k=1:nSamples
    for i = 1:P.Npr
        if(i == 1)
            xt = x(P.Nt+1,:);
        else
            y_sc = zeros(i-1,1);
            for j = 1:i-1
                y_sc(j) = (yPredict(k,j)/yscale-sc_prms(P.Ndr+P.Np-j+1,1))/(sc_prms(P.Ndr+P.Np-j+1,2)-sc_prms(P.Ndr+P.Np-j+1,1));
            end
            xt = [x(P.Nt+i,1:P.Ndr) x(P.Nt+1, P.Ndr+i:P.Np+P.Ndr) y_sc'];
        end
        yPredict(k, i) = svmpredict(y(P.Nt+i-1), xt, svm_model);
    end
    %pmse(k)=norm((yp(k,1:P.Npr)'-y(P.Nt+1:P.Nt+P.Npr))/10);
    P.Nt = P.Nt+1;
end
yPredict = yPredict./yscale;
end

