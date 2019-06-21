function [ dwnSmpcFbeGrad, detailFbeGradient] = calculateDwnGradientFbe(dwnOptimModel, treeData,...
    dwnSmpcFbeResidual, dwnFactorStepModel, optionDwnFbe)
% This function calcualte the dual gradient for the smpc optimisaiton problem for the 
% DWN. This step uses the offline matrices from the factor step   
% 
% syntax 
%
% INPUT 
%   dwnOptimModel     :  smpc optimisation model of the dwn  
%   treeData    :  uncertainty tree structure 
%   dwnFactorStepModel   :  Cost function
%   dualY     :   dual variable 
%   xInitialState     :  x_0 = p; v_{-1} = q; \hat{v}_{-1} = \hat{q};
%
% OUTPUT 
%   dwnSmpcZvar  : X states 
%                : U control/Input
%  details  :
%

nNode = size(treeData.stage, 1); 
lambda = optionDwnFbe.lambda;

dwnSmpcFbeGrad.x = zeros(size(dwnOptimModel.F{1, 1}, 1), nNode);
dwnSmpcFbeGrad.u = zeros(size(dwnOptimModel.G{1, 1}, 1), nNode);

fbeHessianPrimalDir = dwnSmpcFbeGrad;
dwnSmpcFbeHessianDir = calculateDwnHessianDirection(dwnOptimModel, treeData, dwnSmpcFbeResidual,...
    dwnFactorStepModel);

for i = 1:nNode
    fbeHessianPrimalDir.x(:, i) = dwnOptimModel.F{i, 1}*dwnSmpcFbeHessianDir.X(:, i + 1);
    fbeHessianPrimalDir.u(:, i) = dwnOptimModel.G{i, 1}*dwnSmpcFbeHessianDir.U(:, i);
    dwnSmpcFbeGrad.x(:, i) = dwnSmpcFbeResidual.x(:, i) + lambda*dwnOptimModel.F{i, 1}*....
        dwnSmpcFbeHessianDir.X(:, i + 1);
    dwnSmpcFbeGrad.u(:, i) = dwnSmpcFbeResidual.u(:, i) + lambda*dwnOptimModel.G{i, 1}*...
        dwnSmpcFbeHessianDir.U(:, i);
end 

detailFbeGradient.fbeHessianDir = dwnSmpcFbeHessianDir;
detailFbeGradient.fbeHessianPrimalDir = fbeHessianPrimalDir;

end





