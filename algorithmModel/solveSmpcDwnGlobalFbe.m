function [ dwnSmpcZvar, detailsAlgoFbe] = solveSmpcDwnGlobalFbe(dwnOptimModel, dwnFactorStepModel,...
    treeData, dwnOptimCost, optionDwnFbe)
%
% This function is implements the dual global Fbe algorithm to solve
% the sc-mpc of the DWN with soft constraints on state and hard constraits
% on input.
%
% Syntax
%
% INPUT
%   dwnOptimalModel      :  dwn model for the smpc optimisation
%   dwnFactorStepModel   :  offline matrices for factor step
%   treeData             :  uncertainty data
%   dwnOptimCost         :  cost function
%   optionDwnApg         :  options for the dual APG algorithm
%             Termination options: maximum number of steps etc
%             Type of constraints: soft or hard constaints
%             Intial condition and preious control input.
%

nNode = length(treeData.stage);%toal nodes in the tree
nDualx = size(dwnOptimModel.F{1}, 1);
nDualu = size(dwnOptimModel.G{1}, 1);

optionLbfgs.memory = 5;
optionLbfgs.alphaC = 1;

optionLbfgs.lbfgsObject.colLbfgs = 1;
optionLbfgs.lbfgsObject.memLbfgs = 0;
optionLbfgs.lbfgsObject.skipCount = 0;
optionLbfgs.lbfgsObject.matS = zeros((nDualx + nDualu)*nNode, optionLbfgs.memory);
optionLbfgs.lbfgsObject.matY = zeros((nDualx + nDualu)*nNode, optionLbfgs.memory);
optionLbfgs.lbfgsObject.vecInvRho = zeros( optionLbfgs.memory, 1);
optionLbfgs.lbfgsObject.H = 1;


optionProximal.lambda = optionDwnFbe.lambda;
optionProximal.xmin = reshape(dwnOptimModel.xmin, dwnOptimModel.nx, nNode);
optionProximal.xs = reshape(dwnOptimModel.xs, dwnOptimModel.nx, nNode);
optionProximal.xmax = reshape(dwnOptimModel.xmax, dwnOptimModel.nx, nNode);
optionProximal.umin = reshape(dwnOptimModel.umin, dwnOptimModel.nu, nNode);
optionProximal.umax = reshape(dwnOptimModel.umax, dwnOptimModel.nu, nNode);
optionProximal.nx = dwnOptimModel.nx;
optionProximal.nu = dwnOptimModel.nu;
optionProximal.gamma_xbox = dwnOptimModel.gamma_xbox/optionDwnFbe.lambda;
optionProximal.gamma_xs = dwnOptimModel.gamma_xs/optionDwnFbe.lambda;
optionProximal.iter = 200;
optionProximal.constraints = optionDwnFbe.constraints;

detailsAlgoFbe.terminationCriteria = zeros(1,4);

dualVar.x = zeros(nDualx, nNode);
dualVar.u = zeros(nDualu, nNode);
dualVarOld = dualVar;

primalFeasible.x = zeros(nDualx, nNode);
primalFeasible.u = zeros(nDualu, nNode);

uPrev = dwnOptimModel.L*optionDwnFbe.state.v + optionDwnFbe.state.prev_vhat;
optionDwnFbeObjective.previousU = uPrev;

lambdaInitial = optionDwnFbe.lambda;
betaLs = 0.5;

tic
iStep = 1;

while( iStep < optionDwnFbe.steps )
    if(iStep == 1)
        % step 1: calculate the gradient of the conjugate f
        dwnSmpcZvar = calculateDwnSolveStep(dwnOptimModel, treeData, dualVar, dwnFactorStepModel,...
            optionDwnFbe.x, optionDwnFbe.state);
        dwnSmpcGradientZvar = calculateDwnDualGradient(dwnOptimModel, dwnSmpcZvar);
    end
    %
    lineSearch = 1;
    %optionDwnFbe.lambda = lambdaInitial;
    dualVarLBFGDir = dualVarOld;
    dualVarOld = dualVar;
    while( lineSearch )
        optionProximal.lambda = optionDwnFbe.lambda;
        optionProximal.gamma_xbox = dwnOptimModel.gamma_xbox/optionDwnFbe.lambda;
        optionProximal.gamma_xs = dwnOptimModel.gamma_xs/optionDwnFbe.lambda;
        % step 2 : calculate the proximal of g
        [dwnSmpcTvar, proximalCost] = calculateDwnProximalStep(dwnOptimModel, treeData, dualVarOld, dwnSmpcZvar,...
            optionProximal);
        % step 3 : calculate the residual of global Fbe and gradient of the dual Fbe
        dwnSmpcFbeResidual = calculateDwnFbeResidual(dwnOptimModel, dwnSmpcZvar, dwnSmpcTvar);
        [dwnSmpcFbeGrad, ~] = calculateDwnGradientFbe(dwnOptimModel, treeData, dwnSmpcFbeResidual,...
            dwnFactorStepModel, optionDwnFbe);
        
        % calculating the envelop 
        optionDwnFbeObjective.lambda = optionDwnFbe.lambda;
        optionDwnFbeObjective.dualVar = dualVarOld;
        optionDwnFbeObjective.dwnSmpcZvar = dwnSmpcZvar;
        optionDwnFbeObjective.proximalCost = proximalCost;
        optionDwnFbeObjective.fbeResidual = dwnSmpcFbeResidual;
        
        fbeObjDualY = calculateDwnDualEnvelopObjective( dwnOptimCost, treeData, optionDwnFbeObjective);
        detailsAlgoFbe.fbeObjDualY(iStep) = fbeObjDualY;
        
        if (iStep == 1)
            dualVar.x = dualVarOld.x - optionDwnFbe.lambda*dwnSmpcFbeResidual.x;
            dualVar.u = dualVarOld.u - optionDwnFbe.lambda*dwnSmpcFbeResidual.u;
            residual = [dwnSmpcFbeResidual.x; dwnSmpcFbeResidual.u];
            dimResidual = size(residual);
            detailsAlgoFbe.normSmpcResidual(iStep) = norm(reshape(residual, dimResidual(1)*dimResidual(2), 1));
        else
            % calculating the LBFGS direction
            [lbfgsEnvDir, optionLbfgsUpdate] = calcualteDwnLbfgsDirection( dwnSmpcFbeGrad, dwnSmpcFbeGradOld,...
                dualVarOld, dualVarLBFGDir, optionLbfgs);
            
            detailsAlgoFbe.deltaFbeObjDualY(iStep) = fbeObjDualY - detailsAlgoFbe.fbeObjDualY(iStep - 1);
            detailsAlgoFbe.H(iStep) = optionLbfgsUpdate.lbfgsObject.H;
            detailsAlgoFbe.invRho(iStep) = optionLbfgsUpdate.invRho;
            detailsAlgoFbe.skipCount(iStep) = optionLbfgsUpdate.lbfgsObject.skipCount;
            detailsAlgoFbe.direction(iStep) = 0;
            for iNode = 1:nNode
                detailsAlgoFbe.direction(iStep) = detailsAlgoFbe.direction(iStep) +...
                    lbfgsEnvDir.x(:, iNode)'*dwnSmpcFbeGrad.x(:, iNode) + lbfgsEnvDir.u(:, iNode)'*...
                    dwnSmpcFbeGrad.u(:, iNode);
            end
            
            tau = 1;
            beta = 0.5;
            lineSearchIter = 1;
            maxLineSearchIter = 10;
            if((detailsAlgoFbe.direction(iStep)) < 1e-3)
                detailsAlgoFbe.innerLoops(iStep,1) = 0;
                
                while ( lineSearchIter < maxLineSearchIter)
                    dualVarW.x = dualVarOld.x + tau*lbfgsEnvDir.x;
                    dualVarW.u = dualVarOld.u + tau*lbfgsEnvDir.u;
                    
                    % primal x at dual variable w
                    dwnSmpcZvarDualW = calculateDwnSolveStep(dwnOptimModel, treeData, dualVarW, dwnFactorStepModel,...
                        optionDwnFbe.x, optionDwnFbe.state);
                    dwnSmpcGradientZvarDualW = calculateDwnDualGradient(dwnOptimModel, dwnSmpcZvarDualW);
                    % primal z at dual variable w
                    [dwnSmpcTvarDualW, proximalCostDualW] = calculateDwnProximalStep(dwnOptimModel,...
                        treeData, dualVarW, dwnSmpcZvarDualW, optionProximal);
                    % calculate the residual
                    dwnSmpcFbeResidualDualW = calculateDwnFbeResidual(dwnOptimModel, dwnSmpcZvarDualW, dwnSmpcTvarDualW);
                    % calculate the global fbe value
                    optionDwnFbeObjective.dualVar = dualVarW;
                    optionDwnFbeObjective.dwnSmpcZvar = dwnSmpcZvarDualW;
                    optionDwnFbeObjective.proximalCost = proximalCostDualW;
                    optionDwnFbeObjective.fbeResidual = dwnSmpcFbeResidualDualW;
                    fbeObjDualW = calculateDwnDualEnvelopObjective(dwnOptimCost, treeData, optionDwnFbeObjective);
                    detailsAlgoFbe.fbeObjDualW(iStep) = fbeObjDualW;
                    %[fbeObjDualW fbeObjDualY]
                    if fbeObjDualW <= fbeObjDualY
                        tau = beta*tau;
                        lineSearchIter = lineSearchIter + 1;
                    else
                        %tau
                        detailsAlgoFbe.tau(iStep) = tau;
                        lineSearchIter = maxLineSearchIter + 1;
                    end
                end
            else
                lineSearchIter = maxLineSearchIter;
            end
            
            if lineSearchIter == maxLineSearchIter
                detailsAlgoFbe.tau(iStep) = 0;
                %dualVarW = dualVar;
                %dwnSmpcFbeGradOld = dwnSmpcFbeGrad;
                dualVar.x = dualVarOld.x - optionDwnFbe.lambda*dwnSmpcFbeResidual.x;
                dualVar.u = dualVarOld.u - optionDwnFbe.lambda*dwnSmpcFbeResidual.u;
                residual = [dwnSmpcFbeResidual.x; dwnSmpcFbeResidual.u];
            else
                %dwnSmpcFbeGradOld = dwnSmpcFbeGrad;
                dualVar.x = dualVarW.x - optionDwnFbe.lambda*dwnSmpcFbeResidualDualW.x;
                dualVar.u = dualVarW.u - optionDwnFbe.lambda*dwnSmpcFbeResidualDualW.u;
                residual = [dwnSmpcFbeResidualDualW.x; dwnSmpcFbeResidualDualW.u];
            end  
            dimResidual = size(residual);
            detailsAlgoFbe.normSmpcResidual(iStep) = norm(reshape(residual, dimResidual(1)*dimResidual(2), 1));
               
        end
        % calculate the dual gradient at the next step
        dwnSmpcZvarNextIter = calculateDwnSolveStep(dwnOptimModel, treeData, dualVar, dwnFactorStepModel,...
            optionDwnFbe.x, optionDwnFbe.state);
        dwnSmpcGradientZvarNextIter = calculateDwnDualGradient(dwnOptimModel, dwnSmpcZvarNextIter);
        
        if(iStep == 1)
            changeSmpcGrad.x = dwnSmpcGradientZvarNextIter.x - dwnSmpcGradientZvar.x;
            changeSmpcGrad.u = dwnSmpcGradientZvarNextIter.u - dwnSmpcGradientZvar.u;
        else 
            changeSmpcGrad.x = dwnSmpcGradientZvarNextIter.x - dwnSmpcGradientZvarDualW.x;
            changeSmpcGrad.u = dwnSmpcGradientZvarNextIter.u - dwnSmpcGradientZvarDualW.u;
        end 
    
        residualGradient = [changeSmpcGrad.x; changeSmpcGrad.u];
        dimResidualGradient = size(residualGradient);
        normResidualGradient = norm(reshape(residualGradient, dimResidualGradient(1)*dimResidualGradient(2), 1));
        detailsAlgoFbe.ratioResidual(iStep) = normResidualGradient/detailsAlgoFbe.normSmpcResidual(iStep);
        
        if (detailsAlgoFbe.ratioResidual(iStep) < 1)
            lineSearch = 0;
            dwnSmpcZvar = dwnSmpcZvarNextIter;
            dwnSmpcGradientZvar = dwnSmpcGradientZvarNextIter; 
            dwnSmpcFbeGradOld = dwnSmpcFbeGrad;
            
            if(iStep > 1)
                optionLbfgs.lbfgsObject = optionLbfgsUpdate.lbfgsObject;
                optionLbfgs.numeratorH = optionLbfgsUpdate.numeratorH;
                optionLbfgs.denomenatorH = optionLbfgsUpdate.denomenatorH;
                optionLbfgs.invRho = optionLbfgsUpdate.invRho;
            end
            detailsAlgoFbe.lambda(iStep) = optionDwnFbe.lambda;
        else 
            optionDwnFbe.lambda = betaLs*optionDwnFbe.lambda;
        end
    end
    %
    if( mod(iStep, 500) == 0)
        optionDwnFbe.lambda = lambdaInitial;
        %optionProximal.lambda = optionDwnFbe.lambda;
        %optionProximal.gamma_xbox = dwnOptimModel.gamma_xbox/optionDwnFbe.lambda;
        %optionProximal.gamma_xs = dwnOptimModel.gamma_xs/optionDwnFbe.lambda;
    end
    iStep = iStep + 1;
end


%{
while( iStep < optionDwnFbe.steps )
    % step 1: calculate the gradient of the conjugate f
    dwnSmpcZvar = calculateDwnSolveStep(dwnOptimModel, treeData, dualVar, dwnFactorStepModel,...
        optionDwnFbe.x, optionDwnFbe.state);
    dwnSmpcGradientZvar = calculateDwnDualGradient(dwnOptimModel, dwnSmpcZvar);
    % step 2 : calculate the proximal of g
    [dwnSmpcTvar, proximalCost] = calculateDwnProximalStep(dwnOptimModel, treeData, dualVar, dwnSmpcZvar,...
        optionProximal);
    % step 3 : calculate the residual of global Fbe and gradient of the dual Fbe
    dwnSmpcFbeResidual = calculateDwnFbeResidual(dwnOptimModel, dwnSmpcZvar, dwnSmpcTvar);
    [dwnSmpcFbeGrad, ~] = calculateDwnGradientFbe(dwnOptimModel, treeData, dwnSmpcFbeResidual,...
        dwnFactorStepModel, optionDwnFbe);
    % check for direction L-bfgs direction
    %if( iStep == 1)
    if( iStep < optionDwnFbe.steps)
        optionDwnFbeObjective.previousU = uPrev;
        optionDwnFbeObjective.lambda = optionDwnFbe.lambda;
        optionDwnFbeObjective.dualVar = dualVar;
        optionDwnFbeObjective.dwnSmpcZvar = dwnSmpcZvar;
        optionDwnFbeObjective.proximalCost = proximalCost;
        optionDwnFbeObjective.fbeResidual = dwnSmpcFbeResidual;
        
        fbeObjDualY = calculateDwnDualEnvelopObjective( dwnOptimCost, treeData,...
            optionDwnFbeObjective);
        detailsAlgoFbe.fbeObjDualY(iStep) = fbeObjDualY;
        
        dwnSmpcFbeGradOld = dwnSmpcFbeGrad;
        dualVarOld = dualVar;
        dualVar.x = dualVarOld.x - optionDwnFbe.lambda*dwnSmpcFbeResidual.x;
        dualVar.u = dualVarOld.u - optionDwnFbe.lambda*dwnSmpcFbeResidual.u;
        residual = [dwnSmpcFbeResidual.x; dwnSmpcFbeResidual.u];
        dimResidual = size(residual);
        detailsAlgoFbe.normSmpcResidual(iStep) = norm(reshape(residual, dimResidual(1)*dimResidual(2), 1));
        
        if(strcmp(optionDwnFbe.lineSearch, 'yes'))
            dwnSmpcZvarNextIter = calculateDwnSolveStep(dwnOptimModel, treeData, dualVar, dwnFactorStepModel,...
                optionDwnFbe.x, optionDwnFbe.state);
            dwnSmpcGradientZvarNextIter = calculateDwnDualGradient(dwnOptimModel, dwnSmpcZvarNextIter);
            changeSmpcGrad.x = dwnSmpcGradientZvarNextIter.x - dwnSmpcGradientZvar.x;
            changeSmpcGrad.u = dwnSmpcGradientZvarNextIter.u - dwnSmpcGradientZvar.u;
            residualGradient = [changeSmpcGrad.x; changeSmpcGrad.u];
            dimResidualGradient = size(residualGradient);
            normResidualGradient = norm(reshape(residualGradient, dimResidualGradient(1)*dimResidualGradient(2), 1));
            detailsAlgoFbe.ratioResidual(iStep) = normResidualGradient/detailsAlgoFbe.normSmpcResidual(iStep);
        end
        
    else
        [lbfgsEnvDir, optionLbfgsUpdate] = calcualteDwnLbfgsDirection( dwnSmpcFbeGrad, dwnSmpcFbeGradOld,...
            dualVar, dualVarOld, optionLbfgs);
        
        optionDwnFbeObjective.dualVar = dualVar;
        optionDwnFbeObjective.dwnSmpcZvar = dwnSmpcZvar;
        optionDwnFbeObjective.proximalCost = proximalCost;
        optionDwnFbeObjective.fbeResidual = dwnSmpcFbeResidual;
        
        fbeObjDualY = calculateDwnDualEnvelopObjective(dwnOptimCost, treeData, optionDwnFbeObjective);
        
        detailsAlgoFbe.fbeObjDualY(iStep) = fbeObjDualY;
        detailsAlgoFbe.deltaFbeObjDualY(iStep) = fbeObjDualY - detailsAlgoFbe.fbeObjDualY(iStep - 1);
        detailsAlgoFbe.H(iStep) = optionLbfgs.lbfgsObject.H;
        detailsAlgoFbe.direction(iStep) = 0;
        for iNode = 1:nNode
            detailsAlgoFbe.direction(iStep) = detailsAlgoFbe.direction(iStep) +...
                lbfgsEnvDir.x(:, iNode)'*dwnSmpcFbeGrad.x(:, iNode) + lbfgsEnvDir.u(:, iNode)'*...
                dwnSmpcFbeGrad.u(:, iNode);
        end
        
        tau = 1;
        beta = 0.5;
        lineSearchIter = 1;
        maxLineSearchIter = 10;
        if((detailsAlgoFbe.direction(iStep)) < 1e-3)
            detailsAlgoFbe.innerLoops(iStep,1) = 0;
            
            while ( lineSearchIter < maxLineSearchIter)
                dualVarW.x = dualVar.x + tau*lbfgsEnvDir.x;
                dualVarW.u = dualVar.u + tau*lbfgsEnvDir.u;
                
                % primal x at dual variable w
                dwnSmpcZvarDualW = calculateDwnSolveStep(dwnOptimModel, treeData, dualVarW, dwnFactorStepModel,...
                    optionDwnFbe.x, optionDwnFbe.state);
                % primal z at dual variable w
                [dwnSmpcTvarDualW, proximalCostDualW] = calculateDwnProximalStep(dwnOptimModel,...
                    treeData, dualVarW, dwnSmpcZvarDualW, optionProximal);
                % calculate the residual
                dwnSmpcFbeResidualDualW = calculateDwnFbeResidual(dwnOptimModel, dwnSmpcZvarDualW, dwnSmpcTvarDualW);
                % calculate the global fbe value
                optionDwnFbeObjective.dualVar = dualVarW;
                optionDwnFbeObjective.dwnSmpcZvar = dwnSmpcZvarDualW;
                optionDwnFbeObjective.proximalCost = proximalCostDualW;
                optionDwnFbeObjective.fbeResidual = dwnSmpcFbeResidualDualW;
                fbeObjDualW = calculateDwnDualEnvelopObjective(dwnOptimCost, treeData, optionDwnFbeObjective);
                detailsAlgoFbe.fbeObjDualW(iStep) = fbeObjDualW;
                
                if fbeObjDualW <= fbeObjDualY
                    tau = beta*tau;
                    lineSearchIter = lineSearchIter + 1;
                else
                    detailsAlgoFbe.tau(iStep) = tau;
                    lineSearchIter = maxLineSearchIter + 1;
                end
            end
        else
            lineSearchIter = maxLineSearchIter;
        end
         
        if lineSearchIter == maxLineSearchIter
            detailsAlgoFbe.tau(iStep) = 0;
            dualVarW = dualVar;
            dwnSmpcFbeGradOld = dwnSmpcFbeGrad;
            dualVarOld = dualVar;
            dualVar.x = dualVar.x - optionDwnFbe.lambda*dwnSmpcFbeResidual.x;
            dualVar.u = dualVar.u - optionDwnFbe.lambda*dwnSmpcFbeResidual.u;
            residual = [dwnSmpcFbeResidual.x; dwnSmpcFbeResidual.u];
            dimResidual = size(residual);
            detailsAlgoFbe.normSmpcResidual(iStep) = norm(reshape(residual, dimResidual(1)*dimResidual(2), 1));
        else
            dualVarOld = dualVar;
            dwnSmpcFbeGradOld = dwnSmpcFbeGrad;
            dualVar.x = dualVarW.x - optionDwnFbe.lambda*dwnSmpcFbeResidualDualW.x;
            dualVar.u = dualVarW.u - optionDwnFbe.lambda*dwnSmpcFbeResidualDualW.u;
            residual = [dwnSmpcFbeResidualDualW.x; dwnSmpcFbeResidualDualW.u];
            dimResidual = size(residual);
            detailsAlgoFbe.normSmpcResidual(iStep) = norm(reshape(residual, dimResidual(1)*dimResidual(2), 1));
        end
        optionLbfgs.lbfgsObject = optionLbfgsUpdate.lbfgsObject;
        optionLbfgs.numeratorH = optionLbfgsUpdate.numeratorH;
        optionLbfgs.denomenatorH = optionLbfgsUpdate.denomenatorH;
        optionLbfgs.invRho = optionLbfgsUpdate.invRho;
    end
   
    
    if( mod(iStep, 10) == 0)
        optionDwnFbe.lambda = optionDwnFbe.lambda;
        optionProximal.lambda = optionDwnFbe.lambda;
        optionProximal.gamma_xbox = dwnOptimModel.gamma_xbox/optionDwnFbe.lambda;
        optionProximal.gamma_xs = dwnOptimModel.gamma_xs/optionDwnFbe.lambda;
    end
    iStep = iStep + 1;
end
%}

detailsAlgoFbe.gpad_solve = toc;
%detailsAlgoFbe.dualVarW = dualVarW;
detailsAlgoFbe.dualVar = dualVar;
detailsAlgoFbe.dwnSmpcTvar = dwnSmpcTvar;

end







