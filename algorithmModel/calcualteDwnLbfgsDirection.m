function [ lbfgsEnvDir, optionLbfgsUpdate ] = calcualteDwnLbfgsDirection( gradEnv, gradEnvOld, dualVar,...
    dualVarOld, optionLbfgs)
%
% This function calculate the direction using quasi-newton method, limited memory BFGS method.
%
% Syntax : [ lbfgsEnvDir, optionLbfgs ] = calcualteDwnLbfgsDirection( gradEnv, gradEnvOld, dualVar,...
%    dualVarOld, optionLbfgs)
%
% Input  
%   gradEnv     : gradient of the envelope
%   gravEnvOld  : old gradient envelope 
%   dualVar     : current dual variable 
%   dualVarOld  : previous dual variable
% 
% Output     
%   lbfgsEnvDir : direction calculated with LBFGS method
%   optionLbfgs : LBFGS object 
%

lbfgsObject = optionLbfgs.lbfgsObject;
memory = optionLbfgs.memory;
alphaC = optionLbfgs.alphaC;

deltaDualVar = [dualVar.x - dualVarOld.x; dualVar.u - dualVarOld.u];
dimDualVar = size(deltaDualVar);
dimDualXvar = size(dualVar.x);
dimDualUvar = size(dualVar.u);
deltaGradVar = [gradEnv.x - gradEnvOld.x; gradEnv.u - gradEnvOld.u];
dimGradVar = size(deltaGradVar);

sVarLbfgsIter = reshape(deltaDualVar, dimDualVar(1)*dimDualVar(2), 1);
yVarLbfgsIter = reshape(deltaGradVar, dimGradVar(1)*dimGradVar(2), 1);
gradIter = reshape([gradEnv.x;gradEnv.u], dimGradVar(1)*dimGradVar(2), 1);
invRhoVar = sVarLbfgsIter'*yVarLbfgsIter;

if norm(gradIter) < 1,alphaC = 3;end
if invRhoVar/(sVarLbfgsIter'*sVarLbfgsIter) > 1e-6*norm(gradIter) ^alphaC
    lbfgsObject.colLbfgs = 1 + mod(lbfgsObject.colLbfgs, memory);
    lbfgsObject.memLbfgs = min(lbfgsObject.memLbfgs + 1, memory);
    lbfgsObject.matS(:, lbfgsObject.colLbfgs) = sVarLbfgsIter;
    lbfgsObject.matY(:, lbfgsObject.colLbfgs) = yVarLbfgsIter;
    lbfgsObject.vecInvRho(lbfgsObject.colLbfgs)  = invRhoVar;
else
    lbfgsObject.skipCount = lbfgsObject.skipCount + 1;
end

H = invRhoVar/(yVarLbfgsIter'*yVarLbfgsIter);
if(H < 0 || abs(H - lbfgsObject.H) == 0)
    lbfgsObject.H = 1;
else
    lbfgsObject.H = H;
end

optionLbfgsUpdate.numeratorH = invRhoVar;
optionLbfgsUpdate.denomenatorH = yVarLbfgsIter'*yVarLbfgsIter;
optionLbfgsUpdate.invRho = lbfgsObject.vecInvRho(lbfgsObject.colLbfgs); 

lbfgsEnvDirMat = LBFGS(lbfgsObject.matS , lbfgsObject.matY, lbfgsObject.vecInvRho, lbfgsObject.H,...
    -gradIter, int32(lbfgsObject.colLbfgs), int32(lbfgsObject.memLbfgs));

% implemente the two-loop recursive 
mylbfgsEnvDirMat = -gradIter;
lbfgsAlpha = zeros(1, lbfgsObject.memLbfgs);
for iMemory = 1:lbfgsObject.memLbfgs
    iCol = lbfgsObject.colLbfgs - iMemory + 1;
    if(iCol <= 0)
        iCol = iCol + lbfgsObject.memLbfgs;
    end
    lbfgsAlpha(iCol) = lbfgsObject.matS(:, iCol)'*mylbfgsEnvDirMat/lbfgsObject.vecInvRho(iCol);
    mylbfgsEnvDirMat = mylbfgsEnvDirMat - lbfgsAlpha(iCol)*lbfgsObject.matY(:, iCol);
end 
%norm(mylbfgsEnvDirMat)
mylbfgsEnvDirMat = lbfgsObject.H*mylbfgsEnvDirMat;
%norm(mylbfgsEnvDirMat)
for iMemory = lbfgsObject.memLbfgs:-1:1
    iCol = lbfgsObject.colLbfgs - iMemory + 1;
    if(iCol <= 0)
        iCol = iCol + lbfgsObject.memLbfgs;
    end
    beta = lbfgsObject.matY(:, iCol)'*mylbfgsEnvDirMat/lbfgsObject.vecInvRho(iCol);
    ll(iCol) = beta;
    ll(iCol + lbfgsObject.memLbfgs) = lbfgsAlpha(iCol) - beta;
    ll(iCol + 2*lbfgsObject.memLbfgs) = lbfgsObject.matY(:, iCol)'*mylbfgsEnvDirMat;
    mylbfgsEnvDirMat = mylbfgsEnvDirMat + (lbfgsAlpha(iCol) - beta)*lbfgsObject.matS(:, iCol);
end


optionLbfgsUpdate.lbfgsObject = lbfgsObject;
optionLbfgsUpdate.lbfgsEnvDirMat = lbfgsEnvDirMat;
optionLbfgsUpdate.myLbfgsEnvDirMat = mylbfgsEnvDirMat;
optionLbfgsUpdate.lbfgsAlpha = lbfgsAlpha;
lbfgsEnvDirMat = reshape(lbfgsEnvDirMat, dimDualVar(1), dimDualVar(2)); 
lbfgsEnvDir.x = lbfgsEnvDirMat(1:dimDualXvar(1), :);
lbfgsEnvDir.u = lbfgsEnvDirMat(dimDualXvar(1) + 1:dimDualXvar(1) + dimDualUvar(1), :);

end

