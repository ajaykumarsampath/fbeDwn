function [dwnSmpcFbeResidual] = calculateDwnFbeResidual(dwnOptimModel, dwnSmpcZvar, dwnSmpcTvar)
% This function calcualte the residual of the Fbe of the smpc optimisaiton problem for the 
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

nNode = size(dwnSmpcTvar.u, 2); 

dwnSmpcFbeResidual.x = zeros(size(dwnOptimModel.F{1, 1}, 1), nNode);
dwnSmpcFbeResidual.u = zeros(size(dwnOptimModel.G{1, 1}, 1), nNode);

for i = 1:nNode
    dwnSmpcFbeResidual.x(:, i) = dwnSmpcTvar.x(:, i) - dwnOptimModel.F{i, 1}*dwnSmpcZvar.X(:, i+1);
    dwnSmpcFbeResidual.u(:, i) = dwnSmpcTvar.u(:, i) - dwnOptimModel.G{i, 1}*dwnSmpcZvar.U(:, i);
end 

end





