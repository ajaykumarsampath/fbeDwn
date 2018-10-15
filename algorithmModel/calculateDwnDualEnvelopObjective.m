function [fbeObjDualY] = calculateDwnDualEnvelopObjective( dwnOptimCost, treeData, optionDwnFbeObjective)
% 
% This function calculates the objective function for the dual Fbe at the
%   corresponding dual variable
% 
% Syntax 
%  [fbeObjDualY] = calculateDwnDualEnvelopObjective(dwnOptimModel, dwnOptimCost, treeData, dualVar,...
%            dwnSmpcZvar, dwnSmpcTvar, dwnOptimCost)
%
% Input 
%   dwnOptimModel :
%   dwnOptimCost : 
%   treeData :
%   dualVar :
%   dwnSmpcZvar :
%   dwnSmpcYvar :
%   optionDwnFbe :
%
% Output 
%  fbeObjDualY :


fbeObjDualY = 0;
nNodes = size(treeData.stage, 1);

lambda = optionDwnFbeObjective.lambda;
previousU = optionDwnFbeObjective.previousU;
dualVar = optionDwnFbeObjective.dualVar;
dwnSmpcZvar = optionDwnFbeObjective.dwnSmpcZvar;
proximalCost = optionDwnFbeObjective.proximalCost;
fbeResidual = optionDwnFbeObjective.fbeResidual;
dimDualVarX = size(dualVar.x);
dimDualVarU = size(dualVar.u);
dualVarVec = reshape([dualVar.x;dualVar.u], (dimDualVarX(1) + dimDualVarU(1))*dimDualVarX(2), 1);
fbeResidualVec = reshape([fbeResidual.x;fbeResidual.u], (dimDualVarX(1) + dimDualVarU(1))*...
    dimDualVarX(2), 1);

matU = dwnSmpcZvar.U;
for iNode = 1:nNodes
    iStage = treeData.stage(iNode) + 1;
    iProb = treeData.prob(iNode);
    if iNode == 1
        deltaU = matU(:, iNode) - previousU;
    else
        ancestorNode = treeData.ancestor(iNode);
        deltaU = matU(:, iNode) - matU(:, ancestorNode);
    end 
    fbeObjDualY = fbeObjDualY + iProb*deltaU'*dwnOptimCost.Wu*deltaU;
    fbeObjDualY = fbeObjDualY + iProb*dwnOptimCost.alpha(iStage, :)*matU(:, iNode);
end
fbeObjDualY = fbeObjDualY + 0.5*lambda*norm(fbeResidualVec)^2;
fbeObjDualY = fbeObjDualY - dualVarVec'*fbeResidualVec + proximalCost.distanceXset +...
    proximalCost.distanceSafe;

end